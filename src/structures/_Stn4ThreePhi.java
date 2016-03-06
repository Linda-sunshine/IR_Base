package structures;

import java.util.Arrays;

import posteriorRegularization.logisticRegression.pr_test;

public class _Stn4ThreePhi extends _Stn{
	public _Stn4ThreePhi(int index, _SparseFeature[] x, String[] rawTokens, String[] posTags, String rawSource) {
		super(index, x, rawTokens, posTags, rawSource);
	}
	
	public _Stn4ThreePhi(_SparseFeature[] x, String[] rawTokens, String[] posTags, String rawSource, int label) {
		super(x, rawTokens, posTags, rawSource, label);
	}
	
	public void setTopicsVct(int k) {
		m_topics = new double[k+1];

		Arrays.fill(m_topics, 0);
	}

}
