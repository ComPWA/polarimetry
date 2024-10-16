version=1.9.4
if command -v julia >/dev/null 2>&1 && ! julia --version | grep -q "julia version $version"; then
  juliaup add $version &&
    juliaup default $version &&
    julia --project -e '
      using Pkg
      Pkg.add("IJulia")
      import IJulia
      IJulia.installkernel("julia-amplitude-serialization")
    '
fi
