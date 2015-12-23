package clustering;

import cc.mallet.types.FeatureVector;
import structures._Corpus;
import structures._Doc;

/**
 * Overwrite the kmeans algorithm by considering each document as a query and new features.
 * @author lin
 *
 */
public class KMeansAlg4Query extends KMeansAlg{
	int m_dim;
	public KMeansAlg4Query(_Corpus c, int k, int dim) {
		super(c, k);
		m_dim = dim;
	}
	
	/***Overwrite the create instance method
	 * indices: 0 - the number of features.
	 * values: corresponding values. */
	FeatureVector createInstance(_Doc d) {
		d.setQueryDim(m_dim);
		d.setQueryValues();
		int[] indices = d.getQueryIndices();
		for(int i:indices) {
			m_dict.lookupIndex(i, true);
		}
		return new FeatureVector(m_dict, indices, d.getQueryValues());
	} 
}
