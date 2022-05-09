begin
	import ThreeBodyDecay.amplitude
	using ThreeBodyDecay.StaticArrays

	function amplitude(dc::DecayChain, σs, two_λs)
	    k = dc.k; i,j = ij_from_k(k);
	    tbs = dc.tbs
	    s = tbs.ms.m0^2
	    #
	    two_s = dc.two_s
	    two_js = tbs.two_js
	    #
	    itr_two_λs′ = itr(SVector{3}(tbs.two_js[1],tbs.two_js[2],tbs.two_js[3]))
	    f = 0.0
	    for two_τ = -two_s:2:two_s, two_λs′ in itr_two_λs′
	        f += wrong_Zksτ(k,two_s,two_τ,two_λs,two_λs′,σs,tbs) *
	            amplitude(dc.HRk, two_τ, two_λs′[k]) *
	            amplitude(dc.Hij, two_λs′[i], two_λs′[j])
	    end
	    lineshape = dc.Xlineshape(σs[k])
	    return f * lineshape
	end

	function wrong_Zksτ(k, two_s, two_τ, two_λs, two_λs′, σs, tbs; refζs=(1,2,3,1))
	    i,j = ij_from_k(k)
	    #
	    ms² = tbs.ms^2
	    two_js = tbs.two_js
	    two_j0 = two_js[4]
	    two_λ0 = two_λs[4]
	    #
	    w0 = wr(k,refζs[4],0);
	    wi = wr(k,refζs[i],i);
	    wj = wr(k,refζs[j],j);
	    wk = wr(k,refζs[k],k);
	    #
	    cosθ = cosθij(k,σs,ms²)
	    #
	    val =   phase(tbs.two_js[k]-two_λs′[k]) * # particle-2 convention
	        sqrt(two_s+1) * wignerd_doublearg(two_s, two_τ, two_λs′[i]-two_λs′[j], cosθ) *
	            phase(tbs.two_js[j]-two_λs′[j]) # particle-2 convention
	    #
	    val *= two_j0==0 ? (two_τ == two_λs′[k]) : ThreeBodyDecay.wignerd_doublearg_sign(two_j0, two_λ0, two_τ-two_λs′[k], cosζ(w0, σs, ms²), ispositive(w0))
	    val *= two_js[i]==0 ? 1 : ThreeBodyDecay.wignerd_doublearg_sign(two_js[i], two_λs′[i], two_λs[i], cosζ(wi, σs, ms²), !(ispositive(wi)))
	    val *= two_js[j]==0 ? 1 : ThreeBodyDecay.wignerd_doublearg_sign(two_js[j], two_λs′[j], two_λs[j], cosζ(wj, σs, ms²), !(ispositive(wj)))
	    val *= two_js[k]==0 ? 1 : ThreeBodyDecay.wignerd_doublearg_sign(two_js[k], two_λs′[k], two_λs[k], cosζ(wk, σs, ms²), !(ispositive(wk)))
	    #
	    return val
	end
end
