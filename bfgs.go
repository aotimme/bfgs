package bfgs

import (
  "math"
)

type Function func([]float64) float64
type Gradient func([]float64) []float64

func norm(x []float64) (n float64) {
  n = math.Sqrt(inner(x, x))
  return
}

func inner(x, y []float64) (val float64) {
  for i := 0; i < len(x); i++ {
    val += x[i] * y[i]
  }
  return
}

func lineSearch(f Function, g Gradient, x0, dir []float64) (x []float64, alpha float64) {
  c1 := 1e-4
  c2 := 0.9
  beta := 0.5

  n := len(x0)
  grad0 := g(x0)
  fx0 := f(x0)
  dirGrad0 := inner(dir, grad0)
  x = make([]float64, n)

  alpha = 1.0
  for {
    for i := 0; i < n; i++ {
      x[i] = x0[i] + alpha * dir[i]
    }
    if alpha < 1e-9 {
      break
    }
    grad := g(x)
    dirGrad := inner(dir, grad)
    // Wolfe conditions (Armijo rule && curvature condition)
    if f(x) <= fx0 + c1 * alpha * dirGrad0 && dirGrad >= c2 * dirGrad0 {
      break
    }
    alpha *= beta
  }
  return
}

// Mostly taken from http://en.wikipedia.org/wiki/Broyden-Fletcher-Goldfarb-Shanno_algorithm#Algorithm
func Minimize(f Function, g Gradient, x0 []float64) (x []float64, value float64) {
  n := len(x0)

  bkInv := make([][]float64, n)
  for i := 0; i < n; i++ {
    bkInv[i] = make([]float64, n)
    bkInv[i][i] = 1.0
  }

  xk := x0
  gk := g(x0)

  for norm(gk) > 1e-12 {
    // search direction
    pk := make([]float64, n)
    for i := 0; i < n; i++ {
      for j := 0; j < n; j++ {
        pk[i] -= bkInv[i][j] * gk[j]
      }
    }

    xk1, ak := lineSearch(f, g, xk, pk)

    // update amount
    for i := 0; i < n; i++ {
      pk[i] *= ak
    }

    // evaluate at new x
    gk1 := g(xk1)

    // difference from previous x
    yk := make([]float64, n)
    for i := 0; i < n; i++ {
      yk[i] = gk1[i] - gk[i]
    }

    // update bkInv
    skTyk := 0.0
    for i := 0; i < n; i++ {
      skTyk += pk[i] * yk[i]
    }
    ykBkInvyk := 0.0
    for i := 0; i < n; i++ {
      for j := 0; j < n; j++ {
        ykBkInvyk += yk[i] * bkInv[i][j] * yk[j]
      }
    }
    bk1Inv := make([][]float64, n)
    for i := 0; i < n; i++ {
      bk1Inv[i] = make([]float64, n)
    }
    for i := 0; i < n; i++ {
      for j := 0; j < n; j++ {
        bk1Inv[i][j] = bkInv[i][j] + (skTyk + ykBkInvyk) * pk[i] * pk[j] / (skTyk * skTyk)
        for k := 0; k < n; k++ {
          bk1Inv[i][j] -= (bkInv[i][k] * yk[k] * pk[j] + pk[i] * yk[k] * bkInv[k][j]) / skTyk
        }
      }
    }

    // values for next iteration
    bkInv = bk1Inv
    gk = gk1
    xk = xk1
  }

  x = make([]float64, n)
  copy(x, xk)
  value = f(x)

  return
}
