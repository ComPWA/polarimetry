// Author: Alessandro Pilloni, 2021

#include "TString.h"
#include "TComplex.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TRandom3.h"
#include "TTree.h"
#include "TGenPhaseSpace.h"

#include "SpecialFunctions.h"
#define I TComplex::I()
#define CONVPARTICLE2 true
#define MARANFLAG false
#define PLANE 1

const double mK = .493677, mp = 0.938272, mpi = .13957039, mLc = 2.28646;

const TComplex dgamma[6][4][4] = {
//gamma0
        { { 1., 0., 0., 0. }, { 0., 1., 0., 0. }, { 0., 0., -1., 0. }, { 0., 0., 0., -1. } },
//gamma1
        { { 0., 0., 0., 1. }, { 0., 0., 1., 0. }, { 0., -1., 0., 0. }, { -1., 0., 0., 0. } },
//gamma2
        { { 0., 0., 0., -I }, { 0., 0., I, 0. }, { 0., I, 0., 0. }, { -I, 0., 0., 0. } },
//gamma3
        { { 0., 0., 1., 0. }, { 0., 0., 0., -1. }, { -1., 0., 0., 0. }, { 0., 1., 0., 0. } },
//gamma4 = gamma0
        { { 1., 0., 0., 0. }, { 0., 1., 0., 0. }, { 0., 0., -1., 0. }, { 0., 0., 0., -1. } },
//gamma5
        { { 0., 0., 1., 0. }, { 0., 0., 0., 1. }, { 1., 0., 0., 0. }, { 0., 1., 0., 0. } }
};

double g0(int i, int j)
{
	if (i < 0 || i > 3 || j < 0 || j > 3) return 0.;
	if (i < 2 && j >= 2) return -1.;
	if (i >= 2 && j < 2) return -1.;
	return 1.;
}

double cov(TLorentzVector v, int mu, bool covariant = true)
{
	switch (mu)
	{
		case 0: return v.E();
		case 1: return v.Px() * (covariant ? -1 : 1);
		case 2: return v.Py() * (covariant ? -1 : 1);
		case 3: return v.Pz() * (covariant ? -1 : 1);
	}
	return 0.;
}

double lambda1hlf(double a,double b, double c)
{
	return sqrt(a*a + b*b + c*c - 2*a*b - 2*a*c - 2*b*c);
}


void lambdac_covariant(int num = 1, TString filename = "default.root")
{
	TFile *f = new TFile(filename, "RECREATE");
	gRandom->SetSeed(1);
	double masses_Lc[3] = { mp, mK, mpi };
	// double masses_Psi[2] = { 0.1556583745, 0.1556583745 };
	// double masses_Lam[2] = { 0.938272, 0.13957061 };

	// double mup_E, mup_Px, mup_Py, mup_Pz;
	// double mum_E, mum_Px, mum_Py, mum_Pz;
	double p_E, p_Px, p_Py, p_Pz;
	double K_E, K_Px, K_Py, K_Pz;
	double pi_E, pi_Px, pi_Py, pi_Pz;
	double weight, s,t,u, ampl2, interf;
	double ampl2_00, ampl2_01, ampl2_02, ampl2_11, ampl2_12, ampl2_22;
	double hel_00, hel_01, hel_02, hel_11, hel_12, hel_22;
	TGenPhaseSpace ev_Lc;

//	TLorentzVector pB(0., 0., 0., 5.2793);
	TLorentzVector pLc(0., 0., 0., mLc);
	TLorentzVector pK, pP, pPi; //, *pMup, *pMum;
	ev_Lc.SetDecay(pLc, 3, masses_Lc);

	TTree *tt = new TTree("ntuple", "ntuple");
	tt->Branch("weight", &weight);
	// t->Branch("mup_E", &mup_E);
	// t->Branch("mup_Px", &mup_Px);
	// t->Branch("mup_Py", &mup_Py);
	// t->Branch("mup_Pz", &mup_Pz);
	// t->Branch("mum_E", &mum_E);
	// t->Branch("mum_Px", &mum_Px);
	// t->Branch("mum_Py", &mum_Py);
	// t->Branch("mum_Pz", &mum_Pz);
	tt->Branch("p_E", &p_E);
	tt->Branch("p_Px", &p_Px);
	tt->Branch("p_Py", &p_Py);
	tt->Branch("p_Pz", &p_Pz);
	tt->Branch("K_E", &K_E);
	tt->Branch("K_Px", &K_Px);
	tt->Branch("K_Py", &K_Py);
	tt->Branch("K_Pz", &K_Pz);
	tt->Branch("pi_E", &pi_E);
	tt->Branch("pi_Px", &pi_Px);
	tt->Branch("pi_Py", &pi_Py);
	tt->Branch("pi_Pz", &pi_Pz);
	tt->Branch("s", &s);
	tt->Branch("t", &t);
	tt->Branch("u", &u);
	tt->Branch("ampl2", &ampl2);
	tt->Branch("interf", &interf);

	tt->Branch("ampl2_00", &ampl2_00);
	tt->Branch("ampl2_01", &ampl2_01);
	// tt->Branch("ampl2_02", &ampl2_02);
	tt->Branch("ampl2_11", &ampl2_11);
	// tt->Branch("ampl2_12", &ampl2_12);
	// tt->Branch("ampl2_22", &ampl2_22);

	tt->Branch("hel_00", &hel_00);
	tt->Branch("hel_01", &hel_01);
	// tt->Branch("hel_02", &hel_02);
	tt->Branch("hel_11", &hel_11);
	// tt->Branch("hel_12", &hel_12);
	// tt->Branch("hel_22", &hel_22);


	for (int n=0; n < num; n++)
	{
		weight = 1.;
		weight *= ev_Lc.Generate();


		pP = *ev_Lc.GetDecay(0);
		pK = *ev_Lc.GetDecay(1);
		pPi = *ev_Lc.GetDecay(2);
#if PLANE == 1
		double a = -pP.Phi(), b = TMath::Pi() - pP.Theta();
		pP.RotateZ(a); pP.RotateY(b);
		pK.RotateZ(a); pK.RotateY(b);
		pPi.RotateZ(a); pPi.RotateY(b);
		double c = -pK.Phi();
		pK.RotateZ(c); pPi.RotateZ(c);
		pP.Print(); pK.Print(); pPi.Print();

		pP = TLorentzVector(0,0,-0.4622226,1.04594654);
		pK = TLorentzVector(0.36037199, -0.        ,  0.5737196 ,  0.83829537);
		pPi = TLorentzVector(-0.36037199, -0.        , -0.11149701,  0.40221809);

#endif

		// pPsi.Print();
		// pP.Print();
		// pPbar.Print();

		// ev_Psi.SetDecay(*pPsi, 2, masses_Psi);
		// weight *= ev_Psi.Generate();
		// pMup = ev_Psi.GetDecay(0);
		// pMum = ev_Psi.GetDecay(1);

		// mup_E = pMup->E(); mup_Px = pMup->Px(); mup_Py = pMup->Py(); mup_Pz = pMup->Pz();
		// mum_E = pMum->E(); mum_Px = pMum->Px(); mum_Py = pMum->Py(); mum_Pz = pMum->Pz();
		pi_E = pPi.E(); pi_Px = pPi.Px(); pi_Py = pPi.Py(); pi_Pz = pPi.Pz();
		p_E = pP.E(); p_Px = pP.Px(); p_Py = pP.Py(); p_Pz = pP.Pz();
		K_E = pK.E(); K_Px = pK.Px(); K_Py = pK.Py(); K_Pz = pK.Pz();

		// printf("{%.10lf,%.10lf,%.10lf,%.10lf}\n", psi_Px, psi_Py, psi_Pz, psi_E);
		// printf("{%.10lf,%.10lf,%.10lf,%.10lf}\n", p_Px, p_Py, p_Pz, p_E);
		// printf("{%.10lf,%.10lf,%.10lf,%.10lf}\n", pbar_Px, pbar_Py, pbar_Pz, pbar_E);

		TLorentzVector pLambda = pP + pK;
		TLorentzVector pDelta = pPi + pP;
		TLorentzVector pKst = pK + pPi;


		s = pLambda.M2(); t = pDelta.M2(); u = pKst.M2();

        double thetahat_Lambda = pP.Angle(pPi.BoostVector());
        double thetahat_Delta = -pP.Angle(pK.BoostVector());


		TLorentzVector pPi_Lambda = pPi; pPi_Lambda.Boost(-pLambda.BoostVector());
		TLorentzVector pP_Lambda = pP; pP_Lambda.Boost(-pLambda.BoostVector());
		TLorentzVector pK_Lambda = pK; pK_Lambda.Boost(-pLambda.BoostVector());
		TLorentzVector pLc_Lambda = pLc; pLc_Lambda.Boost(-pLambda.BoostVector());
		double theta_Lambda = pP_Lambda.Angle(-pPi_Lambda.BoostVector());

		TLorentzVector pK_Delta = pK; pK_Delta.Boost(-pDelta.BoostVector());
		TLorentzVector pPi_Delta = pPi; pPi_Delta.Boost(-pDelta.BoostVector());
		TLorentzVector pP_Delta = pP; pP_Delta.Boost(-pDelta.BoostVector());
		double theta_Delta = pPi_Delta.Angle(-pK_Delta.BoostVector());


		TLorentzVector pP_Kst = pP; pP_Kst.Boost(-pKst.BoostVector());
		TLorentzVector pK_Kst = pK; pK_Kst.Boost(-pKst.BoostVector());
		TLorentzVector pPi_Kst = pPi; pPi_Kst.Boost(-pKst.BoostVector());
		double theta_Kst = pK_Kst.Angle(-pP_Kst.BoostVector());

		TLorentzVector pDelta_p = pPi; pDelta_p.Boost(-pP.BoostVector());
		TLorentzVector pKst_p = pLc; pKst_p.Boost(-pP.BoostVector());
		TLorentzVector pLambda_p = pK; pLambda_p.Boost(-pP.BoostVector());
		double zetaDelta_p = pDelta_p.Angle(pKst_p.BoostVector());
		// double zetaKst_p = pKst_p.Angle(pLambda_p.BoostVector());
        double zetaLambda_p = pLambda_p.Angle(pKst_p.BoostVector());


        TLorentzVector marP_Lc  = TLorentzVector(-pP.Vect(), pP.E());
        TLorentzVector marLambda_Lc  = TLorentzVector(-pLambda.Vect(), pLambda.E());
        TLorentzVector marP_Lambda = pP; marP_Lambda.Boost(marLambda_Lc.BoostVector());


        marP_Lc.Print(); marP_Lc.BoostVector().Print(); printf("mp %lf\n", marP_Lc.M());
        double gam1 = pP.E() / mp, gam2 = pLambda.E() / pLambda.M(), gam3 = pP_Lambda.E() / mp;
        printf("%lf %lf %lf\n", gam1, gam2, gam3);
        gam1 = 1./sqrt(1. - marP_Lc.BoostVector().Mag2()); gam2 = 1./sqrt(1. - marLambda_Lc.BoostVector().Mag2()); gam3 = 1./sqrt(1. - marP_Lambda.BoostVector().Mag2());
        printf("%lf %lf %lf\n", gam1, gam2, gam3);

        gam1 = (mp*mp + mLc*mLc - u)/(2.*mp*mLc); gam2 = (mLc*mLc - mpi*mpi + s)/(2.*mLc *sqrt(s)); gam3 = (s - mK*mK + mp*mp)/(2.*mp*sqrt(s));
        printf("%lf %lf %lf\n", gam1, gam2, gam3);
        double maran = acos(pow(1. + gam1 + gam2 + gam3, 2)/(1. + gam1)/(1. + gam2)/(1. + gam3) - 1.);

        printf("zeta %lf, maran %lf, thetahat %lf, theta %lf\n", zetaLambda_p, maran, thetahat_Lambda, theta_Lambda);
        printf("zeta %lf marantotal %lf -zeta + marantotal - pi %lf\n", zetaLambda_p, -maran + thetahat_Lambda + theta_Lambda, -zetaLambda_p - maran + thetahat_Lambda + theta_Lambda - TMath::Pi());

    	marP_Lc.Print(); marP_Lc.BoostVector().Print(); printf("mp %lf\n", marP_Lc.M());
        gam1 = pP.E() / mp; gam2 = pDelta.E() / pDelta.M(); gam3 = pP_Delta.E() / mp;
        // printf("%lf %lf %lf\n", gam1, gam2, gam3);
        // gam1 = 1./sqrt(1. - marP_Lc.BoostVector().Mag2()); gam2 = 1./sqrt(1. - marLambda_Lc.BoostVector().Mag2()); gam3 = 1./sqrt(1. - marP_Lambda.BoostVector().Mag2());
        // printf("%lf %lf %lf\n", gam1, gam2, gam3);

        // gam1 = (mp*mp + mLc*mLc - u)/(2.*mp*mLc); gam2 = (mLc*mLc - mpi*mpi + s)/(2.*mLc *sqrt(s)); gam3 = (s - mK*mK + mp*mp)/(2.*mp*sqrt(s));
        // printf("%lf %lf %lf\n", gam1, gam2, gam3);
        maran = acos(pow(1. + gam1 + gam2 + gam3, 2)/(1. + gam1)/(1. + gam2)/(1. + gam3) - 1.);

        printf("zeta %lf, maran %lf, thetahat %lf, theta %lf\n", zetaDelta_p, maran, thetahat_Delta, -(TMath::Pi() - theta_Delta));
        printf("zeta %lf marantotal %lf -zeta + marantotal - pi %lf\n", zetaDelta_p, maran + thetahat_Delta - (TMath::Pi() - theta_Delta), 0.);// -zetaDelta_p + maran + thetahat_Delta + theta_Delta - TMath::Pi());

		// TLorentzVector pKst_K = pPi; pKst_K.Boost(-pK.BoostVector());
		// TLorentzVector pDelta_K = pLc; pDelta_K.Boost(-pK.BoostVector());
		// TLorentzVector pLambda_K = pP; pLambda_K.Boost(-pK.BoostVector());
		// double zetaDelta_K = -pDelta_K.Angle(pLambda_K.BoostVector());
		// double zetaKst_K = -pKst_K.Angle(pLambda_K.BoostVector());

		// TLorentzVector pKst_pi = pK; pKst_pi.Boost(-pPi.BoostVector());
		// TLorentzVector pDelta_pi = pP; pDelta_pi.Boost(-pPi.BoostVector());
		// TLorentzVector pLambda_pi = pB; pLambda_pi.Boost(-pPi.BoostVector());
		// double zetaDelta_pi = -pDelta_pi.Angle(pLambda_pi.BoostVector());
		// double zetaKst_pi = pKst_pi.Angle(pLambda_pi.BoostVector());

		return;


		// ampl2 = 0.; interf = 0.;
		// TComplex amplX, amplPc, amplPcbar;
		// for(int lambda_psi = 1, lambda_psi >= -1, lambda_psi--)
		// for(int lambda_p = 1, lambda_p >= -1, lambda_p-=2)
		// for(int lambda_pbar = 1, lambda_pbar >= -1, lambda_pbar-=2)
		// {

		TComplex pLcpm[4][4], pppm[4][4], prLambda[4][4], amplLambda[4][4], amplKst[4][4];

		for (int i=0; i<4; i++)
		for (int j=0; j<4; j++)
		{
			pLcpm[i][j] = (i == j) ? .5 : 0.;
			pppm[i][j] = (i == j) ? .5 : 0.;;
			prLambda[i][j] = (i == j) ? .5 : 0.;
            amplKst[i][j] = (i == j) ? 1. : 0.;
			for (int mu=0; mu<4; mu++)
			{

				pLcpm[i][j] += dgamma[mu][i][j] * cov(pLc,mu) * .5 / mLc;
				pppm[i][j] += dgamma[mu][i][j] * cov(pP,mu) * .5 / mp;
				prLambda[i][j] += dgamma[mu][i][j] * cov(pLambda,mu) * .5 / sqrt(s);

			}

            amplLambda[i][j] = prLambda[i][j];

		}

		TComplex aXX[3][3];
		for (int i=0; i<3; i++) for (int j=0; j<3; j++) aXX[i][j] = 0.;

		for (int i=0; i<4; i++) for (int j=0; j<4; j++) for (int k=0; k<4; k++) for (int l=0; l<4; l++)
		{
			aXX[0][0] += pppm[i][j]*amplLambda[j][k]*pLcpm[k][l]*(TComplex::Conjugate(amplLambda[i][l]) * g0(i,l));
			aXX[0][1] += pppm[i][j]*amplLambda[j][k]*pLcpm[k][l]*(TComplex::Conjugate(amplKst[i][l]) * g0(i,l));
			aXX[1][0] += pppm[i][j]*amplKst[j][k]*pLcpm[k][l]*(TComplex::Conjugate(amplLambda[i][l]) * g0(i,l));
			aXX[1][1] += pppm[i][j]*amplKst[j][k]*pLcpm[k][l]*(TComplex::Conjugate(amplKst[i][l]) * g0(i,l));
            // aXX[0][0] += pppm[i][j]*amplLambda[j][k]*pLcpm[k][l]*amplLambda[l][i];
			// aXX[0][1] += pppm[i][j]*amplLambda[j][k]*pLcpm[k][l]*amplKst[l][i];
			// aXX[1][0] += pppm[i][j]*amplKst[j][k]*pLcpm[k][l]*amplLambda[l][i];
			// aXX[1][1] += pppm[i][j]*amplKst[j][k]*pLcpm[k][l]*amplKst[l][i];
		}
		// cout << aXX[0][0] << " " << aXX[0][1] << " " << aXX[0][2] << endl;
		// cout << aXX[1][0] << " " << aXX[1][1] << " " << aXX[1][2] << endl;
		// cout << aXX[2][0] << " " << aXX[2][1] << " " << aXX[2][2] << endl;

		// aXX[0][0] = aXX[0][1] = aXX[0][2] = aXX[1][0] = aXX[2][0] = 0.;
		// aXX[1][1] = 1./TComplex(17. - t, -sqrt(17.)*.07).Rho2();
		// aXX[1][2] = 1./TComplex(17. - t, -sqrt(17.)*.07)/TComplex::Conjugate(TComplex(17. - u, -sqrt(17.)*.07));
		// aXX[2][1] = 1./TComplex::Conjugate(TComplex(17. - t, -sqrt(17.)*.07))/TComplex(17. - u, -sqrt(17.)*.07);
		// aXX[2][2] = 1./TComplex(17. - u, -sqrt(17.)*.07).Rho2();

		interf = aXX[0][1] + aXX[0][2] + aXX[1][0] + aXX[2][0];
		ampl2 = interf + aXX[0][0] + aXX[1][1] + aXX[2][2] + aXX[1][2] + aXX[2][1];

		ampl2_00 = aXX[0][0];
		ampl2_01 = aXX[0][1] + aXX[1][0];
		ampl2_02 = aXX[0][2] + aXX[2][0];
		ampl2_11 = aXX[1][1];
		ampl2_12 = aXX[1][2] + aXX[2][1];
		ampl2_22 = aXX[2][2];



		for (int i=0; i<3; i++) for (int j=0; j<3; j++) aXX[i][j] = 0.;
		for (int lambda_p = 1; lambda_p >= -1; lambda_p -= 2) for (int lambda_Lc = 1; lambda_Lc >= -1; lambda_Lc -= 2)
		{

			TComplex aKst = (lambda_p == lambda_Lc) ? 0. : sqrt((pP.E()/mp + 1.)/2.); // H(B->X psi, lambda_psi)
            if (!CONVPARTICLE2) aKst *= lambda_p;

			// if (lambda_psi + lambda_p + lambda_pbar == 6) cout << "z: cos(theta_Pc)" << endl;
			// cout << lambda_pbar << " " << lambda_psi - lambda_p << " " << SpecialFunc::WignerD(1, lambda_pbar, lambda_psi - lambda_p, cos(theta_Pc)) << endl;

			TComplex aLambda = 0.;
			for (int mu_p = 1; mu_p >= -1; mu_p -= 2) for (int mu_Lc = 1; mu_Lc >= -1; mu_Lc -= 2)
			{
				//if (mu_p != lambda_p || mu_Lc != lambda_Lc) continue;

                TComplex bLambda = SpecialFunc::WignerD(1, mu_Lc, mu_p, cos(theta_Lambda)) * TComplex::One();



                //bLambda *= sqrt(pP_Lambda.E() + mp)*sqrt(sqrt(mLc*mLc + pow(pPi.P(),2)) + mLc)/2./sqrt(mp*mLc); // H(B->X psi, lambda_psi)
                bLambda *= sqrt(pP_Lambda.E() + mp)*sqrt(pLc_Lambda.E() + mLc)/2./sqrt(mp*mLc); // H(B->X psi, lambda_psi)
                //if (!CONVPARTICLE2) aX *= lambda_pbar*((lambda_psi == 0) ? -1. : 1);

                if (!MARANFLAG)
                {
				aLambda += bLambda * SpecialFunc::WignerD(1, lambda_Lc, mu_Lc, cos(thetahat_Lambda)) * SpecialFunc::WignerD(1, mu_p, lambda_p, cos(zetaLambda_p))*
				           ((mu_p - lambda_p) % 4 == 0 ? 1. : -1.);
				//if (mu_p == lambda_p && mu_pbar == lambda_pbar)
                }
                else
                {

                aLambda += bLambda * SpecialFunc::WignerD(1, lambda_Lc, mu_Lc, cos(thetahat_Lambda)) * SpecialFunc::WignerD(1, mu_p, lambda_p, cos(maran + thetahat_Lambda + theta_Lambda))*
				           ((mu_p - lambda_p) % 4 == 0 ? 1. : -1.);

                    /* code */
                }

			}

			//printf("%d %lf\n", lambda_psi, aPc.Rho2());


			// aPc /= TComplex(17. - t, -sqrt(17.)*.07);
			// aPcbar /= TComplex(17. - u, -sqrt(17.)*.07);


			aXX[0][0] += aLambda.Rho2();
			aXX[0][1] += (aLambda * TComplex::Conjugate(aKst)).Re();
			aXX[1][0] += (aKst * TComplex::Conjugate(aLambda)).Re();
			aXX[1][1] += aKst.Rho2();


            printf("lp %d lc %d %lf\n", lambda_p, lambda_Lc, aKst.Re());
            printf("lp %d lc %d %lf\n", lambda_p, lambda_Lc, aLambda.Re());

		}

		hel_00 = aXX[0][0];
		hel_01 = aXX[0][1] + aXX[1][0];
		hel_02 = aXX[0][2] + aXX[2][0];
		hel_11 = aXX[1][1];
		hel_12 = aXX[1][2] + aXX[2][1];
		hel_22 = aXX[2][2];


        printf("{ %.12lf, %.12lf, %.12lf, %.12lf };\n", p_E, p_Px, p_Py, p_Pz);
        printf("{ %.12lf, %.12lf, %.12lf, %.12lf };\n", K_E, K_Px, K_Py, K_Pz);
        printf("{ %.12lf, %.12lf, %.12lf, %.12lf };\n", pi_E, pi_Px, pi_Py, pi_Pz);


		// cout << interf << " " << ampl2 << endl;
		tt->Fill();


	}

	tt->Write();


}
