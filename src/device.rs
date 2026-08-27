/// Execution device. Product path is CPU only (Linux/Windows x86_64·aarch64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
}

impl Device {
    #[inline]
    pub fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }
}
