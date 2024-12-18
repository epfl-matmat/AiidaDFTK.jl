module AiidaDFTK
using AtomsBase
using Dates
using DFTK
using DocStringExtensions
using InteractiveUtils
using JLD2
using JSON3
using Logging
using LoggingExtras
using MPI
using Pkg
using PrecompileTools
using TimerOutputs
using Unitful
using UnitfulAtomic

@template METHODS =
"""
$(TYPEDSIGNATURES)

$(DOCSTRING)
"""

include("logging.jl")
include("parse_kwargs.jl")
include("store_hdf5.jl")

# Helper function to check whether we are on the master process
mpi_master(comm=MPI.COMM_WORLD) = (MPI.Init(); MPI.Comm_rank(comm) == 0)

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
    model = model_DFT(system; parse_kwargs(data["model_kwargs"])...)
    PlaneWaveBasis(model;     parse_kwargs(data["basis_kwargs"])...)
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
    runtimeargs    = (; maxtime=Second(get(data["scf"], "maxtime", 60*60*24*366)))
    scfres = self_consistent_field(basis; checkpointargs..., runtimeargs..., kwargs...)

    output_files = [checkpointfile, "self_consistent_field.json"]
    save_scfres("self_consistent_field.json", scfres; save_ψ=false, save_ρ=false)
    save_ψ = get(data["scf"], "save_ψ", false)
    save_scfres(checkpointfile, scfres; save_ψ, save_ρ=true)
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
        results  = getproperty(DFTK, Symbol(funcname))(scfres; kwargs...)

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
function run_json(filename::AbstractString; extra_output_files=String[], min_version=typemin(VersionNumber), max_version=typemax(VersionNumber))
    @info("$LOG_IMPORTS_SUCEEDED --"
        * " This indicates that AiidaDFTK was installed correctly"
        * " and that the MPI environment is likely correct.")

    version = pkgversion(@__MODULE__)
    if min_version <= version && version < max_version
        @info("$LOG_VERSION_OK --"
            * " Expected AiidaDFTK version ∈ [$min_version, $max_version)."
            * " Actual: $(version).")
    else
        msg = ("$LOG_VERSION_MISMATCH --"
            * " Expected AiidaDFTK version ∈ [$min_version, $max_version)."
            * " Actual: $(version).")
        @error msg
        error(msg)
    end

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
    DFTK.setup_threading(; n_blas=1, n_fft=1)
    # Don't modify DFTK threads automatically.
    # They are controlled by a preference, changing it can trigger a recompilation!
    # The best we can do is warn the user.
    if DFTK.get_DFTK_threads() > 1
        @warn("Running through MPI, but threading was not disabled!"
            * " DFTK threads: $(DFTK.get_DFTK_threads()).")
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

    # Run Post SCF routines, but only if SCF converged
    if scfres.converged
        (; output_files) = run_postscf(data, scfres)
        append!(all_output_files, output_files)
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

    @info "$LOG_FINISHED_SUCCESSFULLY."

    (; output_files=all_output_files)
end


"""
Run a DFTK calculation from a json input file. The input file name is expected to be passed
as the first argument when calling Julia (i.e. it should be available via `ARGS`). This
function is expected to be called from queuing system jobscripts, for example:

```bash
julia --project -e 'using AiidaDFTK; AiidaDFTK.run()' /path/to/input/file.json
```

It automatically dumps a logfile `file.log` (i.e. basename of the input file
with the log extension), which contains the log messages (i.e. @info, @warn, ...).
Currently stdout and stderr are still printed.
"""
function run(inputfile; kwargs...)
    # TODO Json logger ?
    logfile = first(splitext(basename(inputfile))) * ".log"
    if mpi_master()
        # Keep logging everything to stderr for manual inspection (AiiDA captures it automatically).
        # Also route log messages to the log file for automatic parsing in AiiDA.
        # Unlike SimpleLogger, FileLogger will always flush. It also truncates the file on creation.
        logger = TeeLogger(current_logger(), FileLogger(logfile))
    else
        logger = NullLogger()
    end

    with_logger(logger) do
        if expanduser("~/.julia") in Pkg.depots()
            @warn("Found ~/.julia in Julia depot path. " *
                "Ensure that you properly specify JULIA_DEPOT_PATH.")
        end
        run_json(inputfile; extra_output_files=[logfile], kwargs...)
    end
end


# Precompilation block with a basic workflow
@setup_workload begin
    # Run a sample silicon input.
    # We don't run any postscf derivative computation in the precompilation workload
    # because of a bad interaction between ForwardDiff and the compilation cache:
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/714
    inputfile = joinpath(@__DIR__, "..", "precompilation_task.json")

    if !PrecompileTools.verbose[]
        # In verbose precompile mode also show the output here
        global_logger(NullLogger())
    end

    mktempdir() do tmpdir
        cd(tmpdir) do
            @compile_workload begin
                run_json(inputfile)
            end
        end
    end
end
end  # module AiidaDFTK
