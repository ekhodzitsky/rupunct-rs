use std::io::Write;

use super::CliError;

pub(super) fn write_help(out: &mut impl Write) -> std::io::Result<()> {
    writeln!(
        out,
        "\
rupunct {}
On-device Russian punctuation and casing restoration. No network.

Usage:
  rupunct [OPTIONS] [TEXT]
  rupunct --jsonl
  echo '{{\"text\":\"привет\"}}' | rupunct

Arguments:
  [TEXT]                 Unpunctuated source (same as --text)

Options:
  -m, --model-dir <DIR>  Weights dir [env: RUPUNCT_MODEL]
                         [default: ~/.rupunct/punct]
      --text <TEXT>      One-shot source (`-` reads stdin)
  -f, --file <PATH>      Read one-shot source from a file (`-` = stdin)
      --jsonl            JSONL worker (default when stdin is a pipe)
      --lossy            On restore failure, emit the original text
      --max-seq <N>      BERT window in tokens [default: 512]
      --overlap <N>      Window overlap in tokens [default: 64]
      --complete <SHELL> Print completions: bash, zsh, or fish
  -h, --help             Print help
  -V, --version          Print version

Exit status:
  0  success
  1  load or restore failure
  2  usage error

JSONL (stdin/stdout, one object per line):
  {{\"ready\":true}}
  → {{\"text\":\"…\"}}           ← {{\"text\":\"…\"}} | {{\"error\":\"…\"}}
  Optional \"id\" on a request is copied onto the response.",
        env!("CARGO_PKG_VERSION")
    )
}

pub(super) fn write_complete(shell: &str, out: &mut impl Write) -> Result<(), CliError> {
    match shell {
        "bash" => write_bash(out).map_err(super::io_err),
        "zsh" => write_zsh(out).map_err(super::io_err),
        "fish" => write_fish(out).map_err(super::io_err),
        other => Err(CliError::usage(format!(
            "unknown shell '{other}' (expected bash, zsh, or fish)"
        ))),
    }
}

fn flags() -> &'static str {
    "--help --version --text --model-dir --file --jsonl --lossy --max-seq --overlap --complete -h -V -m -f"
}

fn write_bash(out: &mut impl Write) -> std::io::Result<()> {
    writeln!(
        out,
        "_rupunct() {{
  local cur
  COMPREPLY=()
  cur=\"${{COMP_WORDS[COMP_CWORD]}}\"
  COMPREPLY=( $(compgen -W \"{}\" -- \"$cur\") )
}}
complete -F _rupunct rupunct",
        flags()
    )
}

fn write_zsh(out: &mut impl Write) -> std::io::Result<()> {
    writeln!(
        out,
        "#compdef rupunct
_arguments -s \\
  '(-h --help)'{{-h,--help}}'[print help]' \\
  '(-V --version)'{{-V,--version}}'[print version]' \\
  '(-m --model-dir)'{{-m,--model-dir}}'[weights dir]:dir:_files -/' \\
  '--text[one-shot source]:text:' \\
  '(-f --file)'{{-f,--file}}'[read source file]:file:_files' \\
  '--jsonl[JSONL worker]' \\
  '--lossy[identity fallback on restore failure]' \\
  '--max-seq[token window]:n:' \\
  '--overlap[token overlap]:n:' \\
  '--complete[shell completions]:shell:(bash zsh fish)' \\
  '*:text:'"
    )
}

fn write_fish(out: &mut impl Write) -> std::io::Result<()> {
    writeln!(
        out,
        "complete -c rupunct -s h -l help -d 'Print help'
complete -c rupunct -s V -l version -d 'Print version'
complete -c rupunct -s m -l model-dir -r -d 'Weights directory'
complete -c rupunct -l text -d 'One-shot source'
complete -c rupunct -s f -l file -r -d 'Read source from file'
complete -c rupunct -l jsonl -d 'JSONL worker'
complete -c rupunct -l lossy -d 'Identity fallback on restore failure'
complete -c rupunct -l max-seq -d 'BERT window in tokens'
complete -c rupunct -l overlap -d 'Window overlap in tokens'
complete -c rupunct -l complete -xa 'bash zsh fish' -d 'Print completions'"
    )
}
