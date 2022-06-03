
const mK = 0.493677
const mπ = 0.13957018
const mp = 0.938272046
const mΣ = 1.18937
const mΛc = 2.28646


const ms = ThreeBodyMasses(m1=mp, m2=mπ, m3=mK, m0=mΛc)

const tbs = ThreeBodySystem(ms, ThreeBodySpins(1, 0, 0; two_h0=1))

const parities = ['+', '-', '-', '±']

