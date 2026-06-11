package bitpack

import "testing"

// mustPanic runs fn and fails unless it panics.
func mustPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

// TestPackPanics covers Pack's argument-validation panics.
func TestPackPanics(t *testing.T) {
	dst := make([]byte, 16)
	src := make([]uint32, BlockSize)
	mustPanic(t, "bits=0", func() { Pack(dst, src, 0) })
	mustPanic(t, "bits=33", func() { Pack(dst, src, 33) })
	// length not a multiple of 128
	mustPanic(t, "bad-len", func() { Pack(dst, make([]uint32, 1), 8) })
}

// TestUnpackPanics covers Unpack's argument-validation panics.
func TestUnpackPanics(t *testing.T) {
	src := make([]byte, 16)
	dst := make([]uint32, BlockSize)
	mustPanic(t, "bits=0", func() { Unpack(dst, src, 0) })
	mustPanic(t, "bits=33", func() { Unpack(dst, src, 33) })
	// length not a multiple of 128
	mustPanic(t, "bad-len", func() { Unpack(make([]uint32, 1), src, 8) })
}

// TestEmptyInputs covers the zero-block short-circuit return in Pack and Unpack
// (no blocks => nothing to do). Empty src/dst is a valid whole number (zero) of
// 128-int blocks.
func TestEmptyInputs(t *testing.T) {
	for bits := 1; bits <= 32; bits++ {
		if n := Pack(nil, nil, bits); n != 0 {
			t.Fatalf("bits=%d: Pack(empty)=%d want 0", bits, n)
		}
		if n := Pack([]byte{}, []uint32{}, bits); n != 0 {
			t.Fatalf("bits=%d: Pack(empty slices)=%d want 0", bits, n)
		}
		// Must not touch dst / panic on bounds.
		Unpack(nil, nil, bits)
		Unpack([]uint32{}, []byte{}, bits)
	}
	// PackedLen of zero ints is zero for every width.
	if PackedLen(0, 7) != 0 {
		t.Fatal("PackedLen(0,7) != 0")
	}
}
