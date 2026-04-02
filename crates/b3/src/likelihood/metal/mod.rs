use anyhow::{Result, anyhow};
use metal::{
	Buffer, CommandBufferRef, CommandQueue, CompileOptions,
	ComputePipelineState, Device, FunctionConstantValues, MTLDataType,
	MTLResourceOptions, MTLSize,
};

use objc::rc::autoreleasepool;

use std::ffi::c_void;
use std::mem;

use super::Calculator;
use crate::{Transitions, parameters::Tree};

type Row = [f32; 4];
type Transition = [f32; 16];

const METAL_SRC: &str = include_str!("kernels.metal");

pub struct MetalLikelihood {
	#[allow(dead_code)]
	device: Device, // kept alive to prevent Metal resource invalidation
	queue: CommandQueue,

	propose_fn: ComputePipelineState,
	copy_projections_fn: ComputePipelineState,
	update_likelihoods_fn: ComputePipelineState,

	leaves: Buffer,
	projections: Buffer,
	projections_backup: Buffer,
	likelihoods: Buffer,
	children: Buffer,
	transitions: Buffer,
	nodes: Buffer,

	pattern_weights: Vec<u32>,

	scales: Buffer,
	scales_backup: Buffer,
	scale_sums: Buffer,
	scale_sums_backup: Buffer,

	num_patterns: u32,
	num_updated_nodes: u32,
}

impl Calculator<4, f64> for MetalLikelihood {
	fn likelihood(
		&mut self,
		tree: &Tree,
		transitions: &Transitions<4, f64>,
	) -> Result<f64> {
		autoreleasepool(|| {
			let (nodes, leaves_end) = tree.nodes_to_update();
			let (nodes, children) = tree.to_lists(&nodes);
			let tms =
				transitions.matrices(&nodes[..nodes.len() - 1]);
			let tms_f32: Vec<[f32; 16]> = tms
				.iter()
				.map(|matrix| {
					let mut out = [0.0f32; 16];
					for (row_idx, row) in
						matrix.iter().enumerate()
					{
						for (col_idx, &val) in
							row.iter().enumerate()
						{
							out[row_idx * 4
								+ col_idx] = val as f32;
						}
					}
					out
				})
				.collect();
			let frequencies_f32: [f32; 4] =
				transitions.frequencies().map(|f| f as f32);
			self.num_updated_nodes = nodes.len() as u32 - 1;
			let root_children = children.last().unwrap();
			let nodes: Vec<_> =
				nodes.iter().map(|n| *n as u32).collect();
			let children: Vec<_> = children
				.iter()
				.flat_map(|&(l, r)| [l as u32, r as u32])
				.collect();

			// fill shared buffers
			unsafe {
				std::ptr::copy_nonoverlapping(
					nodes.as_ptr() as *const u32,
					self.nodes.contents() as *mut u32,
					nodes.len(),
				);
				std::ptr::copy_nonoverlapping(
					children.as_ptr() as *const u32,
					self.children.contents() as *mut u32,
					children.len(),
				);
				std::ptr::copy_nonoverlapping(
					tms_f32.as_ptr() as *const Transition,
					self.transitions.contents()
						as *mut Transition,
					tms_f32.len(),
				);
			}
			let leaves_end = leaves_end as u32; // mut if add leaves_update()
			let internals_start = leaves_end;

			let cmd_buffer = self.queue.new_command_buffer();
			self.update_all(
				leaves_end,
				internals_start,
				&cmd_buffer,
			)?;

			let root = nodes.last().unwrap();
			self.update_likelihoods(
				*root,
				(
					root_children.0 as u32,
					root_children.1 as u32,
				),
				frequencies_f32,
				&cmd_buffer,
			)?;
			cmd_buffer.commit();
			cmd_buffer.wait_until_completed();

			let likelihoods: &[f32] = unsafe {
				std::slice::from_raw_parts(
					self.likelihoods.contents()
						as *const f32,
					self.num_patterns as usize,
				)
			};
			let scale_sums: &[u32] = unsafe {
				std::slice::from_raw_parts(
					self.scale_sums.contents()
						as *const u32,
					self.num_patterns as usize,
				)
			};
			let result: f64 = likelihoods
				.iter()
				.zip(scale_sums)
				.zip(&self.pattern_weights)
				.map(|((&lk, &scale), &weight)| {
					(f64::from(lk) - f64::from(scale))
						* f64::from(weight)
				})
				.sum();

			Ok(result)
		})
	}

	fn accept(&mut self) -> Result<()> {
		autoreleasepool(|| {
			let cmd_buffer = self.queue.new_command_buffer();
			let blit = cmd_buffer.new_blit_command_encoder();
			blit.copy_from_buffer(
				&self.scale_sums,
				0,
				&self.scale_sums_backup,
				0,
				(self.num_patterns as u64)
					* mem::size_of::<u32>() as u64,
			);
			blit.end_encoding();

			self.copy_projections(true, &cmd_buffer)?;

			cmd_buffer.commit();
			cmd_buffer.wait_until_completed();

			self.num_updated_nodes = 0;

			Ok(())
		})
	}

	fn reject(&mut self) -> Result<()> {
		autoreleasepool(|| {
			let cmd_buffer = self.queue.new_command_buffer();
			let blit = cmd_buffer.new_blit_command_encoder();
			blit.copy_from_buffer(
				&self.scale_sums_backup,
				0,
				&self.scale_sums,
				0,
				(self.num_patterns as u64)
					* mem::size_of::<u32>() as u64,
			);
			blit.end_encoding();

			self.copy_projections(false, &cmd_buffer)?;

			cmd_buffer.commit();
			cmd_buffer.wait_until_completed();

			self.num_updated_nodes = 0;

			Ok(())
		})
	}

	fn num_patterns(&self) -> usize {
		self.num_patterns as usize
	}
}

impl MetalLikelihood {
	fn update_all(
		&self,
		leaves_end: u32,
		internals_start: u32,
		cmd_buffer: &CommandBufferRef,
	) -> Result<()> {
		let encoder = cmd_buffer.new_compute_command_encoder();
		encoder.set_compute_pipeline_state(&self.propose_fn);
		// buffers
		encoder.set_buffer(0, Some(&self.leaves), 0);
		encoder.set_buffer(1, Some(&self.projections), 0);
		encoder.set_buffer(2, Some(&self.scales), 0);
		encoder.set_buffer(3, Some(&self.scale_sums), 0);
		encoder.set_buffer(4, Some(&self.nodes), 0);
		encoder.set_buffer(5, Some(&self.children), 0);
		encoder.set_buffer(6, Some(&self.transitions), 0);
		// scalars
		encoder.set_bytes(
			7,
			mem::size_of::<u32>() as u64,
			&self.num_updated_nodes as *const u32 as *const c_void,
		);
		encoder.set_bytes(
			8,
			mem::size_of::<u32>() as u64,
			&leaves_end as *const u32 as *const c_void,
		);
		encoder.set_bytes(
			9,
			mem::size_of::<u32>() as u64,
			&internals_start as *const u32 as *const c_void,
		);
		// configurations
		let num_groups = (self.num_patterns as u64 * 4).div_ceil(64);
		let grid_cfg = MTLSize::new(num_groups, 1, 1);
		let group_cfg = MTLSize::new(64, 1, 1);
		encoder.dispatch_thread_groups(grid_cfg, group_cfg);
		encoder.end_encoding();
		Ok(())
	}
	fn update_likelihoods(
		&self,
		root: u32,
		(left, right): (u32, u32),
		frequencies: Row,
		cmd_buffer: &CommandBufferRef,
	) -> Result<()> {
		let encoder = cmd_buffer.new_compute_command_encoder();
		encoder.set_compute_pipeline_state(&self.update_likelihoods_fn);
		// buffers
		encoder.set_buffer(0, Some(&self.projections), 0);
		encoder.set_buffer(1, Some(&self.likelihoods), 0);
		encoder.set_buffer(2, Some(&self.scales), 0);
		encoder.set_buffer(3, Some(&self.scale_sums), 0);
		// scalars
		encoder.set_bytes(
			4,
			mem::size_of::<u32>() as u64,
			&root as *const u32 as *const c_void,
		);
		encoder.set_bytes(
			5,
			mem::size_of::<u32>() as u64,
			&left as *const u32 as *const c_void,
		);
		encoder.set_bytes(
			6,
			mem::size_of::<u32>() as u64,
			&right as *const u32 as *const c_void,
		);
		encoder.set_bytes(
			7,
			mem::size_of::<Row>() as u64,
			&frequencies as *const Row as *const c_void,
		);
		// configurations
		let num_groups = (self.num_patterns as u64).div_ceil(64);
		let grid_cfg = MTLSize::new(num_groups, 1, 1);
		let group_cfg = MTLSize::new(64, 1, 1);
		encoder.dispatch_thread_groups(grid_cfg, group_cfg);
		encoder.end_encoding();
		Ok(())
	}
	fn copy_projections(
		&self,
		accept: bool,
		cmd_buffer: &CommandBufferRef,
	) -> Result<()> {
		let encoder = cmd_buffer.new_compute_command_encoder();
		encoder.set_compute_pipeline_state(&self.copy_projections_fn);
		// buffers and scalars
		if accept {
			encoder.set_buffer(0, Some(&self.projections), 0);
			encoder.set_buffer(
				1,
				Some(&self.projections_backup),
				0,
			);
			encoder.set_buffer(2, Some(&self.scales), 0);
			encoder.set_buffer(3, Some(&self.scales_backup), 0);
		} else {
			encoder.set_buffer(
				0,
				Some(&self.projections_backup),
				0,
			);
			encoder.set_buffer(1, Some(&self.projections), 0);
			encoder.set_buffer(2, Some(&self.scales_backup), 0);
			encoder.set_buffer(3, Some(&self.scales), 0);
		}
		encoder.set_buffer(4, Some(&self.nodes), 0);
		// configurations
		let num_groups = (self.num_patterns as u64).div_ceil(128);
		let grid_cfg = MTLSize::new(
			num_groups,
			(self.num_updated_nodes + 1) as u64,
			1,
		);
		let group_cfg = MTLSize::new(128, 1, 1);
		encoder.dispatch_thread_groups(grid_cfg, group_cfg);
		encoder.end_encoding();
		Ok(())
	}
	pub fn new(
		pattern_weights: Vec<u32>,
		leaves: Vec<u8>,
		scale_ln: u32,
	) -> Result<Self> {
		// GPU objects
		let device = Device::system_default()
			.ok_or_else(|| anyhow!("Metal GPU not found"))?;
		let queue = device.new_command_queue();

		// scaling parameters
		let scale_threshold = (-(scale_ln as f32)).exp();
		let scale_mult = (scale_ln as f32).exp();

		// scalars
		let num_patterns = pattern_weights.len();
		let num_leaves = leaves.len() / num_patterns;
		let num_internals = num_leaves - 1;
		let num_nodes = num_leaves + num_internals;
		let num_edges = num_internals * 2;

		// buffers
		let leaves = device.new_buffer_with_data(
			leaves.as_ptr() as *const c_void,
			(num_leaves * num_patterns * mem::size_of::<u8>())
				as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let projections = device.new_buffer(
			(num_nodes * num_patterns * mem::size_of::<Row>())
				as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let projections_backup = device.new_buffer(
			(num_nodes * num_patterns * mem::size_of::<Row>())
				as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let likelihoods = device.new_buffer(
			(num_patterns * mem::size_of::<f32>()) as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let children = device.new_buffer(
			(num_edges * mem::size_of::<u32>()) as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let transitions = device.new_buffer(
			(num_edges * mem::size_of::<Transition>()) as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let nodes = device.new_buffer(
			(num_nodes * mem::size_of::<u32>()) as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let zeros = vec![0u8; num_nodes * num_patterns];
		let scales = device.new_buffer_with_data(
			zeros.as_ptr() as *const c_void,
			zeros.len() as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let scales_backup = device.new_buffer_with_data(
			zeros.as_ptr() as *const c_void,
			zeros.len() as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let zeros = vec![0u32; num_patterns];
		let scale_sums = device.new_buffer_with_data(
			zeros.as_ptr() as *const c_void,
			(zeros.len() * mem::size_of::<u32>()) as u64,
			MTLResourceOptions::StorageModeShared,
		);
		let scale_sums_backup = device.new_buffer_with_data(
			zeros.as_ptr() as *const c_void,
			(zeros.len() * mem::size_of::<u32>()) as u64,
			MTLResourceOptions::StorageModeShared,
		);

		// function constants
		let fcv = FunctionConstantValues::new();
		let num_patterns_u32 = num_patterns as u32;
		fcv.set_constant_value_with_name(
			&num_patterns_u32 as *const u32 as *const c_void,
			MTLDataType::UInt,
			"NUM_PATTERNS",
		);
		let num_leaves_u32 = num_leaves as u32;
		fcv.set_constant_value_with_name(
			&num_leaves_u32 as *const u32 as *const c_void,
			MTLDataType::UInt,
			"NUM_LEAVES",
		);
		fcv.set_constant_value_with_name(
			&scale_ln as *const u32 as *const c_void,
			MTLDataType::UInt,
			"SCALE_LN",
		);
		fcv.set_constant_value_with_name(
			&scale_threshold as *const f32 as *const c_void,
			MTLDataType::Float,
			"SCALE_THRESHOLD",
		);
		fcv.set_constant_value_with_name(
			&scale_mult as *const f32 as *const c_void,
			MTLDataType::Float,
			"SCALE_MULT",
		);

		// compile shaders and create pipelines
		let library = device
			.new_library_with_source(
				METAL_SRC,
				&CompileOptions::new(),
			)
			.map_err(|e| anyhow!("Metal compilation error: {e}"))?;
		let propose_function = library
			.get_function("propose", Some(fcv.clone()))
			.map_err(|e| anyhow!(e))?;
		let update_likelihoods_function = library
			.get_function("update_likelihoods", Some(fcv.clone()))
			.map_err(|e| anyhow!(e))?;
		let copy_projections_function = library
			.get_function("copy_projections", Some(fcv))
			.map_err(|e| anyhow!(e))?;

		let propose_fn = device
			.new_compute_pipeline_state_with_function(
				&propose_function,
			)
			.map_err(|e| anyhow!("propose pipeline: {e}"))?;
		let update_likelihoods_fn = device
			.new_compute_pipeline_state_with_function(
				&update_likelihoods_function,
			)
			.map_err(|e| {
				anyhow!("update_likelihoods pipeline: {e}")
			})?;
		let copy_projections_fn = device
			.new_compute_pipeline_state_with_function(
				&copy_projections_function,
			)
			.map_err(|e| {
				anyhow!("copy_projections pipeline: {e}")
			})?;

		Ok(Self {
			device,
			queue,

			propose_fn,
			copy_projections_fn,
			update_likelihoods_fn,

			leaves,
			projections,
			projections_backup,
			likelihoods,
			children,
			transitions,
			nodes,

			pattern_weights,

			scales,
			scales_backup,
			scale_sums,
			scale_sums_backup,

			num_patterns: num_patterns as u32,
			num_updated_nodes: 0,
		})
	}
}
