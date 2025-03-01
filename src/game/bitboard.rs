use std::fmt;
use std::ops;
use std::usize;

pub const BOARD_SIZE: usize = 20;
pub const MAX_IDX: usize = (BOARD_SIZE * (BOARD_SIZE + 1)) - 1;

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Bitboard(pub u128, pub u128, pub u128, pub u128);

impl Bitboard {
    pub fn is_empty(&self) -> bool {
        ((self.0 & u128::MAX >> 92) | self.1 | self.2 | self.3) == 0
    }

    pub fn translate_origin(self, x: usize, y: usize) -> Bitboard {
        assert!(
            x < BOARD_SIZE && y < BOARD_SIZE,
            "Invalid translation of (0,0) to ({x},{y})"
        );
        self << ((BOARD_SIZE + 1) * y + x)
    }

    pub fn rotate_clock(self) -> Bitboard {
        self.flip().mirror_diagonal()
    }

    pub fn rotate_anticlock(self) -> Bitboard {
        self.mirror_diagonal().flip()
    }

    pub fn bit_lookup(&self, bit_idx: usize) -> bool {
        match bit_idx {
            0..=127 => self.3 & 1 << bit_idx != 0,
            128..=255 => self.2 & 1 << (bit_idx - 128) != 0,
            256..=383 => self.1 & 1 << (bit_idx - 2 * 128) != 0,
            384..=MAX_IDX => self.0 & 1 << (bit_idx - 3 * 128) != 0,
            _ => panic!("Bitboard index {bit_idx} out of range [0, {MAX_IDX}]"),
        }
    }

    fn bit_flip(&mut self, bit_idx: usize) {
        match bit_idx {
            0..=127 => self.3 ^= 1 << bit_idx,
            128..=255 => self.2 ^= 1 << (bit_idx - 128),
            256..=383 => self.1 ^= 1 << (bit_idx - 2 * 128),
            384..=MAX_IDX => self.0 ^= 1 << (bit_idx - 3 * 128),
            _ => panic!("Bitboard index {bit_idx} out of range [0, {MAX_IDX}]"),
        }
    }

    pub fn flip(self) -> Bitboard {
        let mut board_flip = Bitboard::default();
        for i in 0..BOARD_SIZE {
            let right_shift = (BOARD_SIZE + 1) * i;
            let left_shift = (BOARD_SIZE + 1) * (BOARD_SIZE - 1 - i);
            board_flip |= ((self >> right_shift) & row_mask()) << left_shift;
        }
        board_flip
    }

    pub fn mirror_diagonal(self) -> Bitboard {
        let mut board_mirror = self;

        for i in 1..BOARD_SIZE {
            for j in 0..=i {
                let idx_upper = (BOARD_SIZE + 1) * j + i;
                let idx_lower = (BOARD_SIZE + 1) * i + j;
                let bit_upper = self.bit_lookup(idx_upper);
                let bit_lower = self.bit_lookup(idx_lower);
                if (bit_upper && !bit_lower) || (bit_lower && !bit_upper) {
                    board_mirror.bit_flip(idx_lower);
                    board_mirror.bit_flip(idx_upper);
                }
            }
        }
        board_mirror
    }

    pub fn dilate_ortho(self) -> Bitboard {
        let left = self >> 1;
        let up = self >> (BOARD_SIZE + 1);
        let right = self << 1;
        let down = self << (BOARD_SIZE + 1);
        (self | left | right | up | down) & separating_bit_mask()
    }

    pub fn dilate_diag(self) -> Bitboard {
        let left_up = self >> (BOARD_SIZE + 2);
        let right_up = self >> BOARD_SIZE;
        let left_down = self << BOARD_SIZE;
        let right_down = self << (BOARD_SIZE + 2);
        (self | left_up | left_down | right_up | right_down) & separating_bit_mask()
    }

    pub fn into_vecs(self) -> [[bool; BOARD_SIZE]; BOARD_SIZE] {
        let mut bitvecs = [[false; BOARD_SIZE]; BOARD_SIZE];
        let mut r = 0usize;
        let mut c = 0usize;
        let mut line_break = 20usize;
        for i in 0..MAX_IDX {
            if i == line_break {
                r += 1;
                c = 0;
                line_break += 21;
            } else {
                bitvecs[r][c] = self.bit_lookup(i);
                c += 1;
            }
        }
        bitvecs
    }

    fn shl_assign(&mut self, lhs: usize) {
        if lhs > 0 {
            self.0 = (self.0 << lhs) | (self.1 >> (128 - lhs));
            self.1 = (self.1 << lhs) | (self.2 >> (128 - lhs));
            self.2 = (self.2 << lhs) | (self.3 >> (128 - lhs));
            self.3 <<= lhs;
        }
    }

    fn shr_assign(&mut self, rhs: usize) {
        if rhs > 0 {
            self.3 = (self.3 >> rhs) | (self.2 << (128 - rhs));
            self.2 = (self.2 >> rhs) | (self.1 << (128 - rhs));
            self.1 = (self.1 >> rhs) | (self.0 << (128 - rhs));
            self.0 >>= rhs;
        }
    }
}

impl ops::Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self::Output {
        Bitboard(!self.0, !self.1, !self.2, !self.3)
    }
}

impl ops::BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Bitboard(
            self.0 & rhs.0,
            self.1 & rhs.1,
            self.2 & rhs.2,
            self.3 & rhs.3,
        )
    }
}

impl ops::BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
        self.1 &= rhs.1;
        self.2 &= rhs.2;
        self.3 &= rhs.3;
    }
}

impl ops::BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Bitboard(
            self.0 | rhs.0,
            self.1 | rhs.1,
            self.2 | rhs.2,
            self.3 | rhs.3,
        )
    }
}

impl ops::BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
        self.1 ^= rhs.1;
        self.2 ^= rhs.2;
        self.3 ^= rhs.3;
    }
}

impl ops::BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Bitboard(
            self.0 ^ rhs.0,
            self.1 ^ rhs.1,
            self.2 ^ rhs.2,
            self.3 ^ rhs.3,
        )
    }
}

impl ops::BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
        self.1 |= rhs.1;
        self.2 |= rhs.2;
        self.3 |= rhs.3;
    }
}

impl ops::Shl<usize> for Bitboard {
    type Output = Self;

    fn shl(self, lhs: usize) -> Self::Output {
        let mut lhs_cntr = lhs;
        let mut shifted = self;
        while lhs_cntr > 127 {
            shifted.shl_assign(127);
            lhs_cntr -= 127;
        }
        shifted.shl_assign(lhs_cntr);
        shifted
    }
}

impl ops::Shr<usize> for Bitboard {
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output {
        let mut rhs_cntr = rhs;
        let mut shifted = self;
        while rhs_cntr > 127 {
            shifted.shr_assign(127);
            rhs_cntr -= 127;
        }
        shifted.shr_assign(rhs_cntr);
        shifted
    }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut board = String::from("  | ");
        let horizontal: String = "-".repeat(3 * BOARD_SIZE + 43);
        let header = (0..BOARD_SIZE)
            .map(|x| format!("{:>2}", x.to_string()))
            .collect::<Vec<String>>()
            .join(" | ");
        board.push_str(&header);
        board.push_str(&format!(" |  s \n{}\n{:>2}", horizontal, 0));

        let mut line_break: usize = 20;
        for i in 0..MAX_IDX {
            if self.bit_lookup(i) {
                board.push_str("|  x ");
            } else {
                board.push_str("|  . ");
            }
            if i == line_break {
                board.push_str(&format!("\n{}\n{:>2}", horizontal, i / BOARD_SIZE));
                line_break += 21;
            }
        }
        board.push_str("|  .");
        write!(f, "{board}")
    }
}

fn row_mask() -> Bitboard {
    Bitboard(
        0,
        0,
        0,
        [vec![1; BOARD_SIZE], vec![0; 1]]
            .concat()
            .iter()
            .rev()
            .fold(0, |acc: u128, bit: &u8| (acc << 1) + u128::from(*bit)),
    )
}

pub fn separating_bit_mask() -> Bitboard {
    let bits: Vec<u8> = (0..BOARD_SIZE).fold(Vec::new(), |acc: Vec<u8>, _| {
        [acc, vec![1; BOARD_SIZE], vec![0; 1]].concat()
    });
    let parse_bits = |acc: u128, bit: &u8| (acc << 1) + u128::from(*bit);

    Bitboard(
        bits[384..].iter().rev().fold(0, parse_bits),
        bits[256..384].iter().rev().fold(0, parse_bits),
        bits[128..256].iter().rev().fold(0, parse_bits),
        bits[0..128].iter().rev().fold(0, parse_bits),
    )
}
