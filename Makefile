.PHONY: all test test-rust test-python build clean cli develop

# Run all tests
all: test

# Run Rust tests
test-rust:
	cargo test -p ad_core

# Build Python bindings in development mode
develop:
	cd ad_py && maturin develop

# Run Python tests (requires develop first)
test-python: develop
	pytest tests/ -v

# Run all tests
test: test-rust test-python

# Run the CLI binary
cli:
	cargo run -p ad_cli

# Build all Rust crates
build:
	cargo build --workspace

# Build release
release:
	cargo build --workspace --release
	cd ad_py && maturin build --release

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/
	rm -rf ad_py/target/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Format code
fmt:
	cargo fmt --all

# Lint
lint:
	cargo clippy --workspace -- -D warnings
