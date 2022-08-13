using PlutoSliderServer

input_dir = joinpath(@__DIR__, "notebooks")
output_dir = joinpath(@__DIR__, "../docs/_static/julia")
PlutoSliderServer.export_directory(input_dir, Export_output_dir=output_dir)
