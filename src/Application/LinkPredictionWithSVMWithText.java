package Application;

import java.util.ArrayList;

import structures._SparseFeature;
import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.MMB._MMBAdaptStruct;

public class LinkPredictionWithSVMWithText extends LinkPredictionWithSVM{

	int m_lmFvSize = 0;
	public LinkPredictionWithSVMWithText(double c, double rho, int lmFv) {
		super(c, rho);
		m_lmFvSize = lmFv;
	}

	@Override
	protected Feature[] constructOneX(_MMBAdaptStruct ui, _MMBAdaptStruct uj){
		ArrayList<Feature> fv = new ArrayList<Feature>();
		double[] mixI = ui.getMixture(), mixJ = uj.getMixture();
		// construct the feature with mixture portion
		for(int k=0; k<mixI.length; k++){
			if(mixI[k] == 0) continue;
			for(int l=0; l<mixJ.length; l++){
				if(mixJ[l] == 0) continue;
				fv.add(new FeatureNode(k*m_kBar+l+1, mixI[k]*mixJ[l]));
			}
		}
		int offset = m_kBar * m_kBar;
		// construct the feature with polynomial kernal
		for(_SparseFeature fi: ui.getUser().getProfile()){
			for(_SparseFeature fj: uj.getUser().getProfile()){
				fv.add(new FeatureNode(offset + fi.getIndex()*m_lmFvSize+fj.getIndex()+1, fi.getValue()*fj.getValue()));
			}
		}
		// construct the feature with (vi - vj)
		_SparseFeature[] pro_i = ui.getUser().getProfile();
		_SparseFeature[] pro_j = uj.getUser().getProfile();
 		int pi = 0, pj = 0;
 		_SparseFeature fi, fj;
		while(pi < pro_i.length && pj < pro_j.length){
			fi = pro_i[pi];
			fj = pro_j[pj];
			if(fi.getIndex() == fj.getIndex()){
				fv.add(new FeatureNode(fi.getIndex()+offset+1, fi.getValue()-fj.getValue()));
				pi++;pj++;
			} else if(fi.getIndex() < fj.getIndex()){
				fv.add(new FeatureNode(fi.getIndex()+offset+1, fi.getValue()));
				pi++;
			} else{
				fv.add(new FeatureNode(fj.getIndex()+offset+1, -fj.getValue()));
				pj++;
			}
		}
		
		if(pi < pro_i.length){
			for(int i=pi; i<pro_i.length; i++){
				fi = pro_i[i];
				fv.add(new FeatureNode(offset+fi.getIndex()+1, fi.getValue()));
			}
		}
		if(pj < pro_j.length){
			for(int i=pj; i<pro_j.length; i++){
				fj = pro_j[i];
				fv.add(new FeatureNode(offset+fj.getIndex()+1, -fj.getValue()));
			}
		}
		Feature[] x = new Feature[fv.size()];
		for(int i=0; i<fv.size(); i++){
			x[i] = fv.get(i);
		}
		return x;
	}
	
	public void buildProfile(){
		_MMBAdaptStruct user;
		for(_AdaptStruct u: m_mmbModel.getUsers()){
			user = (_MMBAdaptStruct) u;
			// build the profile for the user
			user.getUser().buildProfile("lm");
			// normalize the text feature to sum up to 1
			user.getUser().normalizeProfile();
		}
	}
	// we construct user pair as training instances and input into svm for training
	@Override
	public void trainSVM(){
		// train svm model
		buildProfile();
		
		ArrayList<Feature[]> fvsArr = new ArrayList<Feature[]>();
		ArrayList<Double> ysArr = new ArrayList<Double>();
		constructXsYs(fvsArr, ysArr);

		int trainSize = ysArr.size();
		Feature[][] fvs = new Feature[trainSize][];
		double[] ys = new double[trainSize];
		for(int i=0; i<fvsArr.size(); i++){
			fvs[i] = fvsArr.get(i);
			ys[i] = ysArr.get(i);
		}
		
		Problem libProblem = new Problem();
		libProblem.l = trainSize;
		libProblem.x = fvs;
		libProblem.y = ys;
		libProblem.n = m_kBar*m_kBar+m_lmFvSize + 1;
		libProblem.bias = 1;// bias term in liblinear.

		SolverType type = SolverType.L2R_L1LOSS_SVC_DUAL;// solver type: prime
		m_libModel = Linear.train(libProblem, new Parameter(type, m_C, SVM.EPS));

	}
	
}
