
intensity(Ai::AbstractArray, ci::Vector) = sum(abs2, sum(a .* ci) for a in Ai)
