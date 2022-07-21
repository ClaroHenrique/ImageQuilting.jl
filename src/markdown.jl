using GeoStats
using GeoStatsImages
using ImageQuilting

problem = SimulationProblem(CartesianGrid(200,300,45), :facies => Float64, 1)

solver = IQ(
  :facies => (
    trainimg = geostatsimage("Flumy"),
    tilesize = (49,49,14)
  )
)

@time solve(problem, solver)
