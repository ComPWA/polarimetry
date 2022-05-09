// Author: Alessandro Pilloni, 2021

#include "TString.h"
#include "TComplex.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TRandom3.h"
#include "TTree.h"
#include "TGenPhaseSpace.h"

const double mK = .493677, mp = 0.938272, mpi = .13957039, mLc = 2.28646;

double lambda1hlf(double a,double b, double c)
{
	return sqrt(a*a + b*b + c*c - 2*a*b - 2*a*c - 2*b*c);
}


void lambdac_covariant_only_angles(int num = 1, TString filename = "default.root")
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
//		pP = *ev_Lc.GetDecay(0);
//		pK = *ev_Lc.GetDecay(1);
//		pPi = *ev_Lc.GetDecay(2);

		//Need momenta already aligned in the decay plane system for a proper comparison
		//In DM method, it is possible to express the Wigner rotation as a single rotation only inthe decay plane system
		//Otherwise one needs three rotations not necessarily around the same axis
		pP = TLorentzVector(0,0,-0.4622226,1.04594654);
		pK = TLorentzVector(0.36037199, -0.        ,  0.5737196 ,  0.83829537);
		pPi = TLorentzVector(-0.36037199, -0.        , -0.11149701,  0.40221809);

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
		//double zetaDelta_p = pDelta_p.Angle(pLambda_p.BoostVector());
		// double zetaKst_p = pKst_p.Angle(pLambda_p.BoostVector());
        double zetaLambda_p = pLambda_p.Angle(pKst_p.BoostVector());
	      double zetaDelta_p = pDelta_p.Angle(pKst_p.BoostVector()); //DM assumption

	//Angles from DPD paper

	//Resonance helicity angles Eq.(A3)
	Double_t thetahat_Lambda_DPD = acos(( (mLc*mLc + mpi*mpi - s)*(mLc*mLc + mp*mp - u) -2*mLc*mLc*(t-mpi*mpi-mp*mp) )/lambda1hlf(mLc*mLc,mp*mp,u) / lambda1hlf(mLc*mLc,s,mpi*mpi));

	Double_t thetahat_Delta_DPD = -acos(( (mLc*mLc + mp*mp - u)*(mLc*mLc + mK*mK - t) -2*mLc*mLc*(s-mp*mp-mK*mK) )/lambda1hlf(mLc*mLc,mK*mK,t) / lambda1hlf(mLc*mLc,u,mp*mp));

	//Scattering angles, written exploting cyclic permutations Eq.(A1)
	Double_t mq1,mq3,m1,m2,m3;
	mq1=u; mq3=s; m1=mp; m2=mK; m3=mpi;
	Double_t theta_Kst_DPD = acos((2*mq1*(mq3-m1*m1-m2*m2) - (mq1 + m2*m2-m3*m3) * (mLc*mLc - mq1 - m1*m1))/lambda1hlf(mLc*mLc,m1*m1,mq1) / lambda1hlf(mq1,m2*m2,m3*m3));

	mq1=s; mq3=t; m1=mpi; m2=mp; m3=mK;
	Double_t theta_Lambda_DPD = acos((2*mq1*(mq3-m1*m1-m2*m2) - (mq1 + m2*m2-m3*m3) * (mLc*mLc - mq1 - m1*m1))/lambda1hlf(mLc*mLc,m1*m1,mq1) / lambda1hlf(mq1,m2*m2,m3*m3));

	mq1=t; mq3=u; m1=mK; m2=mpi; m3=mp;
	Double_t theta_Delta_DPD = acos((2*mq1*(mq3-m1*m1-m2*m2) - (mq1 + m2*m2-m3*m3) * (mLc*mLc - mq1 - m1*m1))/lambda1hlf(mLc*mLc,m1*m1,mq1) / lambda1hlf(mq1,m2*m2,m3*m3));

	//Wigner angles Eq.(A7)
	Double_t zetaLambda_p_DPD = -acos((2.*mp*mp*(t-mLc*mLc-mK*mK) + (mLc*mLc+mp*mp-u)*(s-mp*mp-mK*mK))/lambda1hlf(mLc*mLc,mp*mp,u)/lambda1hlf(s,mp*mp,mK*mK));

	Double_t zetaDelta_p_DPD = acos((2.*mp*mp*(s-mLc*mLc-mpi*mpi) + (mLc*mLc+mp*mp-u)*(t-mp*mp-mpi*mpi))/lambda1hlf(mLc*mLc,mp*mp,u)/lambda1hlf(t,mp*mp,mpi*mpi));

	//DM method angles
	//- sign added to consider clockwise rotation around decay plane y axis
	Double_t thetahat_Lambda_DM = pLambda.Angle(-pP.Vect());
	Double_t thetahat_Delta_DM = -pDelta.Angle(-pP.Vect());

	Double_t theta_Lambda_DM = pP_Lambda.Angle(-pPi_Lambda.Vect());
	Double_t theta_Delta_DM = -pP_Delta.Angle(-pK_Delta.Vect());
        Double_t theta_Kst_DM = pK_Kst.Angle(-pP_Kst.Vect());

	//Computation of Wigner angles rewritten, but was already fine before
        Double_t gam1 = pP.Gamma();
	Double_t gam2 = pLambda.Gamma();
	Double_t gam3 = pP_Lambda.Gamma();
	Double_t wigner_angle = acos(pow(1. + gam1 + gam2 + gam3, 2)/(1. + gam1)/(1. + gam2)/(1. + gam3) - 1.);
        double maran =              acos(pow(1. + gam1 + gam2 + gam3, 2)/(1. + gam1)/(1. + gam2)/(1. + gam3) - 1.);
        double zetaLambda_p_DM = thetahat_Lambda_DM+theta_Lambda_DM-wigner_angle;

	gam2 = pDelta.Gamma();
	gam3 = pP_Delta.Gamma();
	wigner_angle = acos(pow(1. + gam1 + gam2 + gam3, 2)/(1. + gam1)/(1. + gam2)/(1. + gam3) - 1.);
	cout << "wigner" << wigner_angle <<endl<<endl;
        double zetaDelta_p_DM = thetahat_Delta_DM+theta_Delta_DM+wigner_angle;

	//Output
	cout<<"thetahat_Lambda"<<endl;
	cout<<"This code = "<<thetahat_Lambda<<"  DPD paper = "<<thetahat_Lambda_DPD<<" DM = "<<thetahat_Lambda_DM<<endl<<endl;
	cout<<"thetahat_Delta"<<endl;
	cout<<"This code = "<<thetahat_Delta<<"  DPD paper = "<<thetahat_Delta_DPD<<" DM = "<<thetahat_Delta_DM<<endl<<endl;

	cout<<"theta_Lambda"<<endl;
	cout<<"This code = "<<theta_Lambda<<"  DPD paper = "<<theta_Lambda_DPD<<" DM = "<<theta_Lambda_DM<<endl<<endl;
	cout<<"theta_Delta"<<endl;
	cout<<"This code = "<<theta_Delta<<"  DPD paper = "<<theta_Delta_DPD<<" DM = "<<theta_Delta_DM<<endl<<endl;
	cout<<"theta_Kst"<<endl;
	cout<<"This code = "<<theta_Kst<<"  DPD paper = "<<theta_Kst_DPD<<" DM = "<<theta_Kst_DM<<endl<<endl;

	cout<<"zetaLambda_p"<<endl;
	cout<<"This code = "<<zetaLambda_p<<"  DPD paper = "<<zetaLambda_p_DPD<<" DM = "<<zetaLambda_p_DM<<endl<<endl;

	cout<<"zetaDelta_p"<<endl;
	cout<<"This code = "<<zetaDelta_p<<"  DPD paper = "<<zetaDelta_p_DPD<<" DM = "<<zetaDelta_p_DM<<endl;

	}
}


// Dear Misha,
// From the discussion had today,
// we propose to add the following sentence at L.187, where the polarization systems are defined:

// > The T-odd property of normal polarization does not depend on the reference system used for its definition,
// as studied with $\Lb\to\Lc\mun\bar{\nu}_{\mu}$ simulated decays.

// It has been checked that a null $P_y$ in the helicity system reached from the true \Lb rest frame,
// comparable with the electroweak Lagrangian of the decay, is zero also when rotated to the helicity [...]
