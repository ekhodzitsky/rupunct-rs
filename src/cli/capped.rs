use std::io::{self, BufRead, Read};

pub(super) enum NextLine {
    Eof,
    Ok(String),
    TooLong,
    InvalidUtf8,
}

/// Read one line, capped at `max` bytes **before** storing more.
/// On overflow, consume the rest of the line and return [`NextLine::TooLong`].
pub(super) fn read_capped_line(reader: &mut impl BufRead, max: usize) -> io::Result<NextLine> {
    let mut buf = Vec::new();
    loop {
        let avail = reader.fill_buf()?;
        if avail.is_empty() {
            if buf.is_empty() {
                return Ok(NextLine::Eof);
            }
            return Ok(decode(buf));
        }
        if let Some(i) = avail.iter().position(|&b| b == b'\n') {
            if buf.len() + i > max {
                reader.consume(i + 1);
                return Ok(NextLine::TooLong);
            }
            buf.extend_from_slice(&avail[..i]);
            reader.consume(i + 1);
            return Ok(decode(buf));
        }
        if buf.len() + avail.len() > max {
            skip_rest_of_line(reader)?;
            return Ok(NextLine::TooLong);
        }
        let n = avail.len();
        buf.extend_from_slice(avail);
        reader.consume(n);
    }
}

fn skip_rest_of_line(reader: &mut impl BufRead) -> io::Result<()> {
    loop {
        let avail = reader.fill_buf()?;
        if avail.is_empty() {
            return Ok(());
        }
        if let Some(i) = avail.iter().position(|&b| b == b'\n') {
            reader.consume(i + 1);
            return Ok(());
        }
        let n = avail.len();
        reader.consume(n);
    }
}

fn decode(buf: Vec<u8>) -> NextLine {
    match String::from_utf8(buf) {
        Ok(mut s) => {
            if s.ends_with('\r') {
                s.pop();
            }
            NextLine::Ok(s)
        }
        Err(_) => NextLine::InvalidUtf8,
    }
}

pub(super) fn read_capped_to_string(reader: &mut impl Read, max: usize) -> io::Result<ReadAll> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = reader.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        if buf.len() + n > max {
            return Ok(ReadAll::TooLong);
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    match String::from_utf8(buf) {
        Ok(s) => Ok(ReadAll::Ok(s)),
        Err(_) => Ok(ReadAll::InvalidUtf8),
    }
}

pub(super) enum ReadAll {
    Ok(String),
    TooLong,
    InvalidUtf8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn capped_line_utf8_and_limit() {
        let mut r = Cursor::new(b"hi\n");
        let NextLine::Ok(s) = read_capped_line(&mut r, 8).unwrap() else {
            panic!("ok");
        };
        assert_eq!(s, "hi");
        let mut r = Cursor::new(b"hi\r\n");
        let NextLine::Ok(s) = read_capped_line(&mut r, 8).unwrap() else {
            panic!("crlf");
        };
        assert_eq!(s, "hi");
        let mut r = Cursor::new(&b"abcdefghij\n"[..]);
        assert!(matches!(
            read_capped_line(&mut r, 4).unwrap(),
            NextLine::TooLong
        ));
        let mut r = Cursor::new([0xff, 0xfe, b'\n']);
        assert!(matches!(
            read_capped_line(&mut r, 8).unwrap(),
            NextLine::InvalidUtf8
        ));
        let mut r = Cursor::new(b"abcdefghij");
        assert!(matches!(
            read_capped_line(&mut r, 4).unwrap(),
            NextLine::TooLong
        ));
        let mut r = Cursor::new(vec![b'x'; 16]);
        assert!(matches!(
            read_capped_to_string(&mut r, 8).unwrap(),
            ReadAll::TooLong
        ));
        let mut r = Cursor::new([0xff, 0xfe]);
        assert!(matches!(
            read_capped_to_string(&mut r, 8).unwrap(),
            ReadAll::InvalidUtf8
        ));
        let mut r = Cursor::new(b"ok");
        assert!(matches!(
            read_capped_to_string(&mut r, 8).unwrap(),
            ReadAll::Ok(_)
        ));
        let mut r = Cursor::new(b"");
        assert!(matches!(
            read_capped_line(&mut r, 8).unwrap(),
            NextLine::Eof
        ));
    }
}
