# Numerical experiments
The test cases have been run with Julia v1.10.6. Start julia with the following command:
```bash
julia --project=.
```

Before running the simulations, please activate the project and instantiate, i.e., start Julia as described above and run the following code in the Julia REPL:
```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

## 1D test cases

Run the following command to reproduce the results
```julia
julia> include("1D_2ndO.jl") # takes roughly 1 hour
```

## 2D test cases

Run the following command to reproduce the results
```julia
julia> include("2D_2ndO.jl") # takes several hours
```
