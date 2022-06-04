
function readjson(path)
    f = read(path, String)
    return JSON.parse(f)
end



function ifhyphenaverage(s::String)
    factor = findfirst('-', s) === nothing ? 1 : 2
    eval(Meta.parse(replace(s, '-' => '+'))) / factor
end

function buildchain(key, dict)
    @unpack mass, width, lineshape = dict
    #
    k = Dict('K' => 1, 'D' => 3, 'L' => 2)[first(key)]
    #
    jp_R = str2jp(dict["jp"])
    parity = jp_R.p
    two_j = jp_R.j |> x2
    #
    massval, widthval = ifhyphenaverage.((mass, width)) ./ 1e3
    #
    i, j = ij_from_k(k)
    #
    @unpack two_js = tbs
    #
    reaction_ij = jp_R => (jp(two_js[i] // 2, parities[i]), jp(two_js[j] // 2, parities[j]))
    reaction_Rk(P0) = jp(two_js[0] // 2, P0) => (jp_R, jp(two_js[k] // 2, parities[k]))
    #
    LS = vcat(possible_ls.(reaction_Rk.(('+', '-')))...)
    minLS = first(sort(vcat(LS...); by=x -> x[1]))
    #
    ls = possible_ls(reaction_ij)
    length(ls) != 1 && error("expected the only ls: $(ls)")
    onlyls = first(ls)
    #
    Hij = ParityRecoupling(two_js[i], two_js[j], reaction_ij)
    Xlineshape = eval(
        quote
            $(Symbol(lineshape))(
                (; m=$massval, Γ=$widthval);
                name=$key,
                l=$(onlyls[1]),
                minL=$(minLS[1]),
                m1=$(ms[i]), m2=$(ms[j]), mk=$(ms[k]), m0=$(ms[0]))
        end)
    return (; k, Xlineshape, Hij, two_s=two_j, parity)
end