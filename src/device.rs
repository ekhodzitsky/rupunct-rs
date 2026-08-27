/// Execution device. Inference is CPU only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum Device {
    #[default]
    Cpu,
}

impl Device {
    #[inline]
    pub(crate) fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_cpu() {
        assert!(Device::Cpu.is_cpu());
        assert!(Device::default().is_cpu());
    }
}
