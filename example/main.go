package main

import (
  "fmt"

  "github.com/aotimme/bfgs"
)

func sqSumFunction(x []float64) float64 {
  val := 0.0
  for _, xx := range x {
    val += xx
  }
  return val * val
}

func sqSumGradient(x []float64) []float64 {
  n := len(x)
  grad := make([]float64, n)
  sum := 0.0
  for i := 0; i < n; i++ {
    sum += x[i]
  }
  for i := 0; i < n; i++ {
    grad[i] = 2 * sum * x[i]
  }
  return grad
}

func main() {
  n := 5
  x0 := make([]float64, n)
  for i := 0; i < n; i++ {
    x0[i] = float64(i)
  }
  x, val := bfgs.Minimize(sqSumFunction, sqSumGradient, x0)
  fmt.Printf("x = %v\n", x)
  fmt.Printf("val = %v\n", val)
}
