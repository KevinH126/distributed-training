// Your test file. Nothing here is graded. Write your own tests as you
// go. Delete or change the examples below.
//
// Useful commands:
//
//	go test .                    // run every test in this folder
//	go test -run TestSeq .       // run only tests whose name has TestSeq
//	go test -v .                 // print each test name as it runs
//	go test -race .              // catch data races (needed for Parts 2 and 3)
//
// A good habit: when you find a bug, add a test that would have
// caught it. Then you never chase the same bug twice.
package main

import "testing"

// A smoke test. When RunSeq is implemented, this passes: the returned
// slice has the right length. It does not check that the model
// actually trains.
func TestExampleSeq(t *testing.T) {
	cfg := SeqConfig{Seed: 1, Steps: 5, BatchSize: 32, LR: 0.01}
	w := RunSeq(cfg)
	if len(w) != D {
		t.Fatalf("RunSeq returned %d weights, want %d", len(w), D)
	}
}

// Template for Part 2. Uncomment when RunLocal works.
func TestExampleLocal(t *testing.T) {
	cfg := LocalConfig{Seed: 1, Steps: 5, BatchSize: 32, LR: 0.01, NRanks: 4}
	w := RunLocal(cfg)
	if len(w) != D {
		t.Fatalf("RunLocal returned %d weights, want %d", len(w), D)
	}
}

func TestLocalMatchesReference(t *testing.T) {
	cfg := LocalConfig{
		Seed:      123,
		Steps:     10,
		BatchSize: 64,
		LR:        0.02,
		NRanks:    4,
	}

	// Build expected result by doing exactly the same math,
	// but without goroutines.
	want := Init(cfg.Seed)
	b := NewBatcher(cfg.Seed, cfg.BatchSize)
	sliceSize := cfg.BatchSize / cfg.NRanks

	for step := 0; step < cfg.Steps; step++ {
		batch := b.Next(step)
		total := make([]float64, D)

		for rank := 0; rank < cfg.NRanks; rank++ {
			start := rank * sliceSize
			end := start + sliceSize

			g := Grad(want, batch[start:end])

			for j := 0; j < D; j++ {
				total[j] += g[j]
			}
		}

		for j := 0; j < D; j++ {
			want[j] -= cfg.LR * total[j] / float64(cfg.BatchSize)
		}
	}

	got := RunLocal(cfg)

	if len(got) != len(want) {
		t.Fatalf("RunLocal returned %d weights, want %d", len(got), len(want))
	}

	for i := 0; i < D; i++ {
		if got[i] != want[i] {
			t.Fatalf(
				"weight %d differs: got %.17g, want %.17g",
				i, got[i], want[i],
			)
		}
	}
}

// With one rank, RunLocal should behave exactly like RunSeq.
func TestLocalOneRankMatchesSeq(t *testing.T) {
	localCfg := LocalConfig{
		Seed:      42,
		Steps:     10,
		BatchSize: 32,
		LR:        0.01,
		NRanks:    1,
	}

	seqCfg := SeqConfig{
		Seed:      localCfg.Seed,
		Steps:     localCfg.Steps,
		BatchSize: localCfg.BatchSize,
		LR:        localCfg.LR,
	}

	got := RunLocal(localCfg)
	want := RunSeq(seqCfg)

	for i := 0; i < D; i++ {
		if got[i] != want[i] {
			t.Fatalf(
				"weight %d differs: RunLocal=%.17g RunSeq=%.17g",
				i, got[i], want[i],
			)
		}
	}
}

// Check that several valid rank counts work correctly.
func TestLocalDifferentRankCounts(t *testing.T) {
	rankCounts := []int{1, 2, 4, 8}

	for _, nRanks := range rankCounts {
		cfg := LocalConfig{
			Seed:      77,
			Steps:     5,
			BatchSize: 64,
			LR:        0.01,
			NRanks:    nRanks,
		}

		want := Init(cfg.Seed)
		b := NewBatcher(cfg.Seed, cfg.BatchSize)
		sliceSize := cfg.BatchSize / cfg.NRanks

		for step := 0; step < cfg.Steps; step++ {
			batch := b.Next(step)
			total := make([]float64, D)

			for rank := 0; rank < cfg.NRanks; rank++ {
				start := rank * sliceSize
				end := start + sliceSize

				g := Grad(want, batch[start:end])

				for j := 0; j < D; j++ {
					total[j] += g[j]
				}
			}

			for j := 0; j < D; j++ {
				want[j] -= cfg.LR * total[j] / float64(cfg.BatchSize)
			}
		}

		got := RunLocal(cfg)

		for i := 0; i < D; i++ {
			if got[i] != want[i] {
				t.Fatalf(
					"NRanks=%d weight %d differs: got %.17g, want %.17g",
					nRanks, i, got[i], want[i],
				)
			}
		}
	}
}

// Same inputs should always produce exactly the same result.
func TestLocalDeterministic(t *testing.T) {
	cfg := LocalConfig{
		Seed:      99,
		Steps:     10,
		BatchSize: 64,
		LR:        0.02,
		NRanks:    4,
	}

	first := RunLocal(cfg)
	second := RunLocal(cfg)

	for i := 0; i < D; i++ {
		if first[i] != second[i] {
			t.Fatalf(
				"RunLocal is not deterministic at weight %d: %.17g vs %.17g",
				i, first[i], second[i],
			)
		}
	}
}

// Template for Part 3. Uncomment when RunWorker works. Starts one
// worker and shuts it down.
//
// func TestExampleWorker(t *testing.T) {
// 	addr, stop, err := RunWorker("localhost:0")
// 	if err != nil {
// 		t.Fatalf("RunWorker: %v", err)
// 	}
// 	defer stop()
// 	if addr == "" {
// 		t.Fatal("RunWorker returned an empty address")
// 	}
// }
