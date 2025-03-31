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
using Pkg.Versions
using PrecompileTools
using TimerOutputs
using Unitful
using UnitfulAtomic

public run

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
        magnetic_moment = convert(Float64, get(atom, "magnetic_moment", 0.0))
        Atom(symbol, position; magnetic_moment)
    end

    bounding_box = convert(Vector{Vector{Float64}},
                           data["periodic_system"]["bounding_box"]) * u"bohr"
    periodic_system(atoms, bounding_box)
end

function build_pseudopotentials(pseudo_data, system)
    kwargs = parse_kwargs(get(pseudo_data, Symbol("\$kwargs"), Dict()))
    family = Dict{Symbol, String}(k => pseudo_data[k]
                                  for k in keys(pseudo_data)
                                  if k != Symbol("\$kwargs"))
    load_psp(family, system; kwargs...)
end

function build_basis(data, system)
    pseudopotentials = build_pseudopotentials(data["pseudopotentials"], system)
    parsed = DFTK.parse_system(system, pseudopotentials)
    model = model_DFT(parsed.lattice, parsed.atoms, parsed.positions;
                      parsed.magnetic_moments,
                      parse_kwargs(data["model_kwargs"])...)
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
    runtimeargs    = (; maxtime=Second(get(data["scf"], "maxtime", 60*60*24*366)))
    scfres = self_consistent_field(basis; checkpointargs..., runtimeargs..., kwargs...)

    save_scfres("self_consistent_field.json", scfres; save_ψ=false, save_ρ=false)
    save_ψ = get(data["scf"], "save_ψ", false)
    save_scfres(checkpointfile, scfres; save_ψ, save_ρ=true)
    scfres
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
    postscf_calcs = data["postscf"]
    for calc in postscf_calcs
        funcname = calc["\$function"]
        kwargs   = parse_kwargs(get(calc, "\$kwargs", Dict()))
        results  = getproperty(DFTK, Symbol(funcname))(scfres; kwargs...)

        store_hdf5(funcname * ".hdf5", (; funcname, results))
    end
end

function run_refinement(data, scfres)
    haskey(data, :refinement) || return

    @info "Running refinement..."
    # First: run a new SCF without any temperature
    # Construct model and basis without temperature
    model = Model(scfres.basis.model; smearing=Smearing.None(), temperature=0)
    basis = PlaneWaveBasis(model; parse_kwargs(data["basis_kwargs"])..., )
    # Run another scf using the starting density from scfres
    interpolations = Dict("basis" => basis, "model" => basis.model)
    kwargs = parse_kwargs(data["scf"]["\$kwargs"]; interpolations)
    runtimeargs = (; maxtime=Second(get(data["scf"], "maxtime", 60*60*24*366)))
    @info "Running SCF again without temperature"
    scfres = self_consistent_field(basis; ρ=scfres.ρ, runtimeargs..., kwargs...)

    refinement_data = data["refinement"]
    # Build larger basis
    basis_ref = PlaneWaveBasis(scfres.basis.model;
                               parse_kwargs(data["basis_kwargs"])...,
                               parse_kwargs(refinement_data["basis_kwargs"])...)
    extra_kwargs = parse_kwargs(get(refinement_data, "\$kwargs", Dict()))
    # Run refinement
    refinement = refine_scfres(scfres, basis_ref; extra_kwargs...)
    # Refine quantities of interest using Q(P) + dQ*ΔP
    (E, dE) = refine_energies(refinement)
    (F, dF) = refine_forces(refinement)
    EplusdE = E.values .+ dE.values
    FplusdF = F .+ dF
    # Refine quantities of interest using Q(P+ΔP)
    ψ_refined = [DFTK.ortho_qr(ψk + δψk) for (ψk, δψk) in zip(refinement.ψ, refinement.δψ)]
    ρ_refined = compute_density(basis_ref, ψ_refined, refinement.occupation)
    E_refined = DFTK.energy(basis_ref, ψ_refined, refinement.occupation; ρ=ρ_refined).energies
    F_refined = compute_forces(basis_ref, ψ_refined, refinement.occupation; ρ=ρ_refined)

    written_energies(E) = [sum(E), E...]

    output = (;
        E_terms = ["total", E.keys...],
        E_base = written_energies(E.values),
        F_base = F,
        E_refined_old = written_energies(EplusdE),
        F_refined_old = FplusdF,
        E_refined_new = written_energies(E_refined.values),
        F_refined_new = F_refined,
    )

    if mpi_master()
        open("refinement.json", "w") do io
            JSON3.pretty(io, (; refinement_kwargs=extra_kwargs, output...))
        end
    end
end


"""
Run a DFTK calculation after the necessary environment has been setup.
"""
function _run(; inputfile::AbstractString, allowed_versions::AbstractString)
    @info("$LOG_IMPORTS_SUCEEDED --"
        * " This indicates that AiidaDFTK was installed correctly"
        * " and that the MPI environment is likely correct.")

    version = pkgversion(@__MODULE__)
    version_spec = allowed_versions == "*" ? VersionSpec() : semver_spec(allowed_versions)
    if version ∈ version_spec
        @info("$LOG_VERSION_OK --"
            * " Expected AiidaDFTK version ∈ $version_spec."
            * " Actual: $(version).")
    else
        error("$LOG_VERSION_MISMATCH --"
            * " Expected AiidaDFTK version ∈ $version_spec."
            * " Actual: $(version).")
    end

    if mpi_master()
        data = open(inputfile, "r") do io
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
    scfres = run_scf(data, system, basis)

    # Run Post SCF routines, but only if SCF converged
    if scfres.converged
        run_postscf(data, scfres)
        run_refinement(data, scfres)
    end

    # Dump timings
    timingfile = "timings.json"
    if mpi_master()
        println(DFTK.timer)
        open(timingfile, "w") do io
            JSON3.pretty(io, TimerOutputs.todict(DFTK.timer))
        end
    end

    @info "$LOG_FINISHED_SUCCESSFULLY."
end


"""
Run a DFTK calculation from a json input file. This function is expected
to be called from queuing system jobscripts, for example:

```bash
julia --project -e 'using AiidaDFTK; AiidaDFTK.run(inputfile="/path/to/input/file.json")'
```

It automatically dumps a logfile `file.log` (i.e. basename of the input file
with the log extension), which contains the log messages (i.e. @info, @warn, ...).
Currently stdout and stderr are still printed.

Required keyword arguments:
- `inputfile`: The input file name.

Optional keyword arguments:
- `allowed_versions`: A range of supported AiidaDFTK versions, in the format supported by Pkg.
"""
function run(; inputfile::AbstractString, allowed_versions::AbstractString="*")
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
        try
            _run(; inputfile, allowed_versions)
        catch e
            @error "Failed because of an exception" exception=(e, catch_backtrace())
            rethrow()
        end
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
                run(; inputfile)
            end
        end
    end
end
end  # module AiidaDFTK
