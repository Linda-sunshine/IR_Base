package structures;

import java.util.ArrayList;
import java.util.HashMap;

import utils.Utils;

public class _Item {
	protected String m_itemID;
	protected ArrayList<_Review> m_reviews;
	protected _SparseFeature[] m_BoWProfile; //The BoW representation of a item.
	protected HashMap<Integer, Double> m_feature;//hashmap representation of item profile
	protected double[] m_itemWeights; // the learned eta from ETBIR
	protected double m_delta = 0.01;
	protected double m_D = 0; //length of all reviews

	public _Item(String id){
		m_itemID = id;
		m_reviews = new ArrayList<_Review>();
	}

	public String getID(){ return m_itemID; }

	public double getLength(){ return this.m_D; }

	public HashMap<Integer, Double> getFeature(){ return m_feature; }
	
	public void addOneReview(_Review r){
		m_reviews.add(r);
	}
	
	// build the profile for the user
	public void buildProfile(String model){
		ArrayList<_SparseFeature[]> reviews = new ArrayList<_SparseFeature[]>();
		m_feature = new HashMap<>();
		m_D = 0;

		if(model.equals("lm")){
			for(_Review r: m_reviews){
				reviews.add(r.getLMSparse());
			}
			m_BoWProfile = Utils.MergeSpVcts(reviews);
		} else{
			for(_Review r: m_reviews){
				reviews.add(r.getSparse());
				m_D += r.getTotalDocLength();
			}
			m_BoWProfile = Utils.MergeSpVcts(reviews);
			m_feature = Utils.MergeFeatureMap(reviews);
		}
	}

	//build language model for item
	public double calcAdditiveSmoothedProb(int index){
		// additive smoothing to smooth a unigram language model
		if(m_feature.containsKey(index)) {
			return (m_feature.get(index) + m_delta) /
					(m_D + m_delta * m_feature.size());
		} else{
			return m_delta / (m_D + m_delta * m_feature.size());
		}
	}
	
	public void normalizeProfile(){
		double sum = 0;
		for(_SparseFeature fv: m_BoWProfile){
			sum += fv.getValue();
		}
		for(_SparseFeature fv: m_BoWProfile){
			double val = fv.getValue() / sum;
			fv.setValue(val);
		}
	}
	
	public _SparseFeature[] getBoWProfile(){
		return m_BoWProfile;
	}
	
	public double[] getItemWeights(){
		return m_itemWeights;
	}
	
	public void setItemWeights(double[] ws){
		m_itemWeights = ws;
	}
		
}
