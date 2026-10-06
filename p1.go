// p1.go -- the file you edit for Project 1.
//
// The file has three parts, one for each part of the project.
// Read project1-desc.pdf for what each part must do.
//
// Rules:
//   - Do not change the names of the functions in this file.
//   - Do not change the arguments they take or what they return.
//   - Do not change the fields inside SeqConfig, LocalConfig, or
//     DistConfig.
//   - You may add your own helper functions and types anywhere.
//   - Do not modify sealed.go or yours_test.go.
//
// After you finish a part, you can run it by hand:
//   go run . -mode seq
//   go run . -mode local -n 4
//   go run . -mode worker -addr localhost:9001
//   go run . -mode coord  -workers localhost:9001,localhost:9002

package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/rpc"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
	// You will need more packages as you go. Add them here when you need
	// them. Likely: "encoding/gob", "errors", "net", "net/rpc", "sync",
	// "sync/atomic".
)

/* Part 1 -- Sequential SGD (20 points). See project1-desc.pdf, Part 1. */

// SeqConfig holds the inputs to RunSeq. Do not add or remove fields.
type SeqConfig struct {
	Seed      int64
	Steps     int
	BatchSize int
	LR        float64

	// OnStep may be set by the test code. If it is not nil, call it
	// after every step with a fresh copy of the weights.
	OnStep func(step int, w []float64)
}

// RunSeq trains for cfg.Steps steps and returns the final weights.
func RunSeq(cfg SeqConfig) []float64 {

	w := Init(cfg.Seed)
	b := NewBatcher(cfg.Seed, cfg.BatchSize)

	for step := 0; step < cfg.Steps; step++ {
		batch := b.Next(step)
		g := Grad(w, batch)
		for i := 0; i < len(w); i++ {
			w[i] -= cfg.LR * g[i] / float64(cfg.BatchSize)
		}
		if cfg.OnStep != nil {
			w_copy := make([]float64, len(w))
			copy(w_copy, w)
			cfg.OnStep(step, w_copy)
		}
	}

	return w

}

/* Part 2 -- Parallel local trainer (15 points). See project1-desc.pdf, Part 2. */

// LocalConfig holds the inputs to RunLocal. NRanks must divide
// BatchSize. Do not add or remove fields.
type LocalConfig struct {
	Seed      int64
	Steps     int
	BatchSize int
	LR        float64
	NRanks    int
}

type Work struct {
	Step  int
	Slice []Example
}

// RunLocal trains using NRanks goroutines and returns the final weights.
func RunLocal(cfg LocalConfig) []float64 {

	w := Init(cfg.Seed)
	if cfg.Steps == 0 {
		return w
	}
	b := NewBatcher(cfg.Seed, cfg.BatchSize)
	work_chans := make([]chan Work, cfg.NRanks)
	grad_chans := make([]chan []float64, cfg.NRanks)
	finalWeights := make(chan []float64)
	SliceSize := cfg.BatchSize / cfg.NRanks

	for rank := 0; rank < cfg.NRanks; rank++ {
		work_chans[rank] = make(chan Work)
		grad_chans[rank] = make(chan []float64)
		rankW := make([]float64, len(w))
		copy(rankW, w)

		go func(r int, weights []float64) {
			for true {
				work := <-work_chans[r]

				g := Grad(weights, work.Slice)

				if r == 0 {
					for i := 1; i < cfg.NRanks; i++ {
						grad_i := <-grad_chans[i]
						for j := range 16 {
							g[j] += grad_i[j]
						}
					}
					for i := 1; i < cfg.NRanks; i++ {
						gCopy := make([]float64, len(g))
						copy(gCopy, g)
						grad_chans[i] <- gCopy
					}
				} else {
					grad_chans[r] <- g
					grad_msg := <-grad_chans[r]
					g = grad_msg
				}

				for i := range weights {
					weights[i] -= cfg.LR * g[i] / float64(cfg.BatchSize)
				}

				if work.Step == cfg.Steps-1 {
					if r == 0 {
						wCopy := make([]float64, len(weights))
						copy(wCopy, weights)
						finalWeights <- wCopy
					}
					break
				}
			}
		}(rank, rankW)
	}

	for step := range cfg.Steps {
		batch := b.Next(step)
		for r := range cfg.NRanks {
			slice := batch[r*SliceSize : ((r + 1) * SliceSize)]
			work_chans[r] <- Work{Step: step, Slice: slice}
		}
	}
	return <-finalWeights
}

/* Part 3 -- Coordinator and workers over RPC (65 points). */

// Worker is the RPC service. You register it under the name "Worker"
// so clients call "Worker.Grad" and "Worker.Ping".
type Worker struct{}

// Grad returns the summed gradient for one slice of a batch.
func (Worker) Grad(args GradArgs, reply *GradReply) error {
	if len(args.W) != D {
		return errors.New("len(w) != 16")
	}
	if len(args.Batch) == 0 {
		return errors.New("len(batch) == 0")
	}

	reply.Grad = Grad(args.W, args.Batch)
	return nil
}

// Ping is a health check. Leave it as it is.
func (Worker) Ping(_ PingArgs, reply *PingReply) error {
	reply.OK = true
	return nil
}

// RunWorker starts a worker on addr and returns its actual address
// plus a stop function that shuts it down.
func RunWorker(addr string) (actualAddr string, stop func(), err error) {
	server := rpc.NewServer()
	worker := &Worker{}
	server.Register(worker)

	listener, err := net.Listen("tcp", addr)

	if err != nil {
		return "", nil, errors.New("Failed to initialize listener")
	}
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}

			go server.ServeConn(conn)
		}
	}()
	var once sync.Once
	stop_func := func() {
		once.Do(func() {
			listener.Close()
		})
	}
	return listener.Addr().String(), stop_func, nil
}

// DistConfig holds the inputs to RunCoordinator. Do not add or remove
// fields.
type DistConfig struct {
	Seed      int64
	Steps     int
	BatchSize int
	LR        float64
	NSlices   int
	Workers   []string
	Timeout   time.Duration
	OnStep    func(step int, w []float64)
}

// RunCoordinator drives training end to end and returns the final
// weights.
func RunCoordinator(cfg DistConfig) []float64 {
	// Write your code here.
	// See project1-desc.pdf, Part 3.2 and Part 3.3.
	// You are free to add helper types and helper functions.
	panic("not implemented yet: Part 3 (RunCoordinator)")
}

/* main -- one binary, four modes. Leave this as it is. */

func main() {
	mode := flag.String("mode", "", "seq | local | worker | coord")
	seed := flag.Int64("seed", 2026, "seed")
	steps := flag.Int("steps", 300, "training steps")
	bs := flag.Int("bs", 64, "batch size")
	lr := flag.Float64("lr", 0.02, "learning rate")
	n := flag.Int("n", 4, "number of ranks (local) or slices (coord)")
	addr := flag.String("addr", "localhost:0", "worker listen address")
	workers := flag.String("workers", "", "comma-separated host:port list (coord)")
	timeout := flag.Duration("timeout", 500*time.Millisecond, "per-attempt timeout")
	flag.Parse()

	switch *mode {
	case "seq":
		eval := NewBatcher(*seed, 1024).Next(0)
		w := RunSeq(SeqConfig{Seed: *seed, Steps: *steps, BatchSize: *bs, LR: *lr})
		fmt.Printf("final loss = %.6f\n", Loss(w, eval))
	case "local":
		eval := NewBatcher(*seed, 1024).Next(0)
		w := RunLocal(LocalConfig{Seed: *seed, Steps: *steps,
			BatchSize: *bs, LR: *lr, NRanks: *n})
		fmt.Printf("final loss = %.6f  (%d ranks)\n", Loss(w, eval), *n)
	case "worker":
		actual, stop, err := RunWorker(*addr)
		if err != nil {
			log.Fatalf("worker start: %v", err)
		}
		fmt.Printf("worker listening on %s\n", actual)
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		stop()
	case "coord":
		if *workers == "" {
			log.Fatal("coord: provide -workers")
		}
		addrs := strings.Split(*workers, ",")
		eval := NewBatcher(*seed, 1024).Next(0)
		w := RunCoordinator(DistConfig{
			Seed: *seed, Steps: *steps, BatchSize: *bs, LR: *lr,
			NSlices: *n, Workers: addrs, Timeout: *timeout,
		})
		fmt.Printf("final loss = %.6f\n", Loss(w, eval))
	default:
		fmt.Fprintln(os.Stderr, "usage: p1 -mode {seq|local|worker|coord} [flags]")
		flag.PrintDefaults()
		os.Exit(2)
	}
}
