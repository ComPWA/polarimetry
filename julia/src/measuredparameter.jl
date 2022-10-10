struct MeasuredParameter
    val::Float64
    stat::Float64
    model::Float64
    syst::Float64
end
MeasuredParameter(val, stat, model=0.0, syst=0.0) =
    MeasuredParameter(val, stat, model, syst)
MeasuredParameter(str::AbstractString) =
    MeasuredParameter(Meta.parse.(split(str, "±"))...)
