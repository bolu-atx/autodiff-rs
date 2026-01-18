//! Metal GPU Backend for ad_tensor (macOS only).
//!
//! This backend uses Apple's Metal framework for GPU-accelerated tensor operations.
//! It provides significant speedups for large tensors and matrix operations.

#[cfg(target_os = "macos")]
mod metal_impl;

#[cfg(target_os = "macos")]
pub use metal_impl::*;

#[cfg(not(target_os = "macos"))]
compile_error!("ad_backend_metal only supports macOS");
