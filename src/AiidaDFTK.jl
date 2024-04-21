module AiidaDFTK
using Logging
using AtomsBase
using DFTK
using DocStringExtensions
using InteractiveUtils
using JLD2
using JSON3
using TimerOutputs
using MPI
using Unitful
using UnitfulAtomic
using Pkg
using Dates

export run_json

export build_system
export build_basis

@template METHODS =
"""
$(TYPEDSIGNATURES)

$(DOCSTRING)
"""

include("parse_kwargs.jl")
include("store_hdf5.jl")

# Helper function to check whether we are on the master process
mpi_master(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_rank(comm) == 0)
mpi_nprocs(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_size(comm))

function build_system(data)
    atoms = map(data["periodic_system"]["atoms"]) do atom
        symbol   = Symbol(atom["symbol"])
        position = convert(Vector{Float64}, atom["position"]) * u"bohr"
        pseudopotential        = atom["pseudopotential"]
        pseudopotential_kwargs = parse_kwargs(get(atom, "pseudopotential_kwargs", Dict()))
        magnetic_moment = convert(Float64, get(atom, "magnetic_moment", 0.0))
        Atom(symbol, position; pseudopotential, pseudopotential_kwargs, magnetic_moment)
    end

    bounding_box = convert(Vector{Vector{Float64}},
                           data["periodic_system"]["bounding_box"]) * u"bohr"
    periodic_system(atoms, bounding_box)
end

function build_basis(data, system)
    model_kwargs = parse_kwargs(data["model_kwargs"])
    xc = pop!(model_kwargs, :xc)
    model = model_DFT(system, xc; model_kwargs...)
    PlaneWaveBasis(model; parse_kwargs(data["basis_kwargs"])...)
end

function run_geometry_optimisation(data, system, basis)
    error("not implemented yet")
end

function run_self_consistent_field(data, system, basis)
    interpolations = Dict("basis" => basis, "model" => basis.model)
    kwargs = parse_kwargs(data["scf"]["\$kwargs"]; interpolations)

    ρ = guess_density(basis, system)
    checkpointfile = data["scf"]["checkpointfile"]
    checkpointargs = kwargs_scf_checkpoints(basis; filename=checkpointfile, ρ)
    maxtime = get(data["scf"], "maxtime", nothing)
    maxtime = maxtime !== nothing ? Second(maxtime) : Year(1)
    scfres = self_consistent_field(basis; checkpointargs..., kwargs..., maxtime)

    output_files = [checkpointfile, "self_consistent_field.json"]
    save_ψ = get(data["scf"], "save_ψ", false)
    save_scfres(checkpointfile, scfres; save_ψ, save_ρ=true)
    save_scfres("self_consistent_field.json", scfres; save_ψ=false, save_ρ=false) #output json after jld2, facilitating checking if checkpoint saved correctly
    (; scfres, output_files)
end

function run_scf(data, system, basis)
    funcname = data["scf"]["\$function"]
    if funcname == "self_consistent_field"
        return run_self_consistent_field(data, system, basis)
    elseif funcname == "geometry_optimisation"
        return run_geometry_optimisation(data, system, basis)
    else
        error("Unknown scf function: $funcname")
    end
end

function run_postscf(data, scfres)
    output_files = String[]
    postscf_calcs = data["postscf"]
    for calc in postscf_calcs
        funcname = calc["\$function"]
        kwargs   = parse_kwargs(get(calc, "\$kwargs", Dict()))

        if funcname == "compute_bands"
            kpath = pop!(kwargs, :kpath, nothing)
            if kpath !== nothing
                kpath = convert(Vector{Vector{Float64}}, kpath)
            else
                error("kpath is not provided for compute_bands")
            end
            kpath = ExplicitKpoints(kpath)
            bands = getproperty(DFTK, Symbol(funcname))(scfres, kpath; kwargs...)
            results = (kpath=kpath.kcoords,eigenvalues=bands.eigenvalues,occupation=bands.occupation)
        else
            results  = getproperty(DFTK, Symbol(funcname))(scfres; kwargs...)
        end

        store_hdf5(funcname * ".hdf5", (; funcname, results))
        push!(output_files, funcname * ".hdf5")
    end
    (; output_files)
end


"""
Run a DFTK calculation from a json input file.
Output is by default written to `stdout` and `stderr`.
The list of generated output files is returned.
"""
function run_json(filename::AbstractString; extra_output_files=String[])
    all_output_files = copy(extra_output_files)

    if mpi_master()
        data = open(filename, "r") do io
            JSON3.read(io)
        end
    else
        data = nothing
    end
    data = MPI.bcast(data, MPI.COMM_WORLD)

    # Print key information about Julia and DFTK
    if mpi_master()
        InteractiveUtils.versioninfo()
        println()
        DFTK.versioninfo()
        println()
    end

    # Threading setup ... maybe later need to take parameters
    # from the JSON into account
    if mpi_nprocs() > 1
        disable_threading()
    else
        setup_threading()
    end

    DFTK.reset_timer!(DFTK.timer)
    system = build_system(data)
    basis  = build_basis(data, system)

    # Print key information about computational setup
    if mpi_master()
        show(stdout, "text/plain", basis)
        println()
    end

    # Run SCF routine
    (; scfres, output_files) = run_scf(data, system, basis)
    append!(all_output_files, output_files)

    # Run Post SCF routines only after SCF converged
    if isfile("self_consistent_field.json")
        scf_json = JSON3.read(read("self_consistent_field.json", String))
        if get(scf_json, "converged", false)
            (; output_files) = run_postscf(data, scfres)
            append!(all_output_files, output_files)
        end
    end

    # Dump timings
    timingfile = "timings.json"
    if mpi_master()
        println(DFTK.timer)
        open(timingfile, "w") do io
            JSON3.pretty(io, TimerOutputs.todict(DFTK.timer))
        end
    end
    push!(all_output_files, timingfile)

    (; output_files=all_output_files)
end


"""
Run a DFTK calculation from a json input file. The input file name is expected to be passed
as the first argument when calling Julia (i.e. it should be available via `ARGS`. This
function is expected to be called from queuing system jobscripts, for example:

```bash
julia --project -e 'using AiidaDFTK; AiidaDFTK.run()' /path/to/input/file.json
```

It automatically dumps a logfile `file.log` (i.e. basename of the input file
with the log extension), which contains the log messages (i.e. @info, @warn, ...).
Currently stdout  and stderr is still printed.
"""
function run()
    # TODO Json logger ?
    inputfile = only(ARGS)
    logfile   = first(splitext(basename(inputfile))) * ".log"
    if mpi_master()
        global_logger(SimpleLogger(open(logfile, "w")))
    else
        global_logger(NullLogger())
    end

    if expanduser("~/.julia") in Pkg.depots()
        @warn("Found ~/.julia in Julia depot path. " *
              "Ensure that you properly specify JULIA_DEPOT_PATH.")
    end
    run_json(inputfile; extra_output_files=[logfile])
end

end
