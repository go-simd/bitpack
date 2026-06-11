//go:build ignore

// Part of the pack_s390x.s generator (run with pack_s390x_gen.go). Emits the
// unpack kernels unpackBits{bits}_VX, the inverse of the pack kernels. See
// pack_s390x_gen.go for the layout and the big-endian VPERM byte-reversal notes.
//
// Unpack schedule for width `bits`: integer k occupies per-lane bits
// [k*bits, k*bits+bits). With word w=(k*bits)/32 and off=(k*bits)%32, the value
// is (W[w]>>off) | (W[w+1]<<(32-off) when it straddles), masked to `bits`. Each
// kernel is a branch-free unrolled sequence of VL+VPERM/VESRLF/VESLF/VO/VN/
// VPERM+VST.
package main

import (
	"fmt"

	"github.com/go-asmgen/asmgen/s390x"
)

// genUnpackZ emits unpackBits{bits}_VX. R2=dst, R3=src, R4=blocks.
func (s *zState) genUnpackZ(bits int, maskSym string) {
	name := fmt.Sprintf("unpackBits%d_VX", bits)
	loop, done := "loop_"+name, "done_"+name
	fn := s390x.NewFunc(name, zSig(), 0)
	fn.LoadArg("dst_base", "R2").LoadArg("src_base", "R3").LoadArg("blocks", "R4").
		Raw("MOVD $%s(SB), R5", s.selSym).
		Raw("VL (R5), V10"). // selrev
		Raw("MOVD $%s(SB), R5", maskSym).
		Raw("VL (R5), V11"). // mask
		Raw("CMPBEQ R4, $0, %s", done).
		Label(loop)

	s.emitUnpackBodyZ(fn, bits)

	fn.Raw("ADD $%d, R3", 16*bits). // bits words * 16 bytes
		Raw("ADD $512, R2").        // 128 ints * 4 bytes
		Raw("ADD $-1, R4").Raw("CMPBNE R4, $0, %s", loop).
		Label(done).Ret()
	s.f.Add(fn.Func())
}

// emitUnpackBodyZ mirrors the amd64 SSE emitUnpackBody.
//
//	Cur = V3   current source word (loaded + reversed)
//	Nxt = V4   next source word (for straddling integers)
//	V   = V1   extracted value being assembled
//	scratch V6 for the straddle merge.
func (s *zState) emitUnpackBodyZ(fn *s390x.Builder, bits int) {
	const cur, nxt, v = "V3", "V4", "V1"
	loaded := -1

	ensure := func(w int) {
		if loaded != w {
			zLoadWord(fn, cur, w)
			loaded = w
		}
	}

	for k := 0; k < 32; k++ {
		start := k * bits
		word := start / 32
		off := start % 32
		end := off + bits

		ensure(word)
		if off == 0 {
			fn.Raw("VLR %s, %s", cur, v) // V = Cur
		} else {
			fn.Raw("VESRLF $%d, %s, %s", off, cur, v) // V = Cur >> off
		}
		if end > 32 {
			zLoadWord(fn, nxt, word+1)
			fn.Raw("VESLF $%d, %s, V6", 32-off, nxt). // V6 = Nxt << (32-off)
				Raw("VO V6, %s, %s", v, v).            // V |= V6
				Raw("VLR %s, %s", nxt, cur)             // pre-cache next as current
			loaded = word + 1
		}
		fn.Raw("VN V11, %s, %s", v, v) // V &= mask
		zStoreVec(fn, v, k)
	}
}

// zLoadWord loads source word w (byte offset 16w) into reg and byte-reverses
// each word so reg holds register-correct uint32 values.
func zLoadWord(fn *s390x.Builder, reg string, w int) {
	if w == 0 {
		fn.Raw("VL (R3), %s", reg)
	} else {
		fn.Raw("MOVD $%d, R6", 16*w).
			Raw("VL (R3)(R6*1), %s", reg)
	}
	fn.Raw("VPERM %s, %s, V10, %s", reg, reg, reg)
}

// zStoreVec stores reg to output vector k (dst[4k..4k+3], byte offset 16k). The
// dst []uint32 array is native (big-endian) on s390x, so a plain VST of the
// register-correct word elements writes the correct values -- no byte reversal.
// Reversal happened on load (zLoadWord) when reading the little-endian packed
// stream.
func zStoreVec(fn *s390x.Builder, reg string, k int) {
	if k == 0 {
		fn.Raw("VST %s, (R2)", reg)
		return
	}
	fn.Raw("MOVD $%d, R6", 16*k).
		Raw("VST %s, (R2)(R6*1)", reg)
}
