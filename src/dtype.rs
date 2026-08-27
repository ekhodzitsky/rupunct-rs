/// Element type. Spike focuses on F32; Q8 storage lands with QMatMul port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    #[default]
    F32,
    I64,
    /// Logical marker for packed Q8_0 weights (not a dense element type yet).
    Q8_0,
}
