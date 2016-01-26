package structures;

public class L2RParameter extends Parameter {
	
	public int m_topK;
	public double m_noiseRatio;
	public double m_lambda4L2R;
	public double m_shrinkage;
	public double m_stepSize;
	public double m_windowSize;
	public double m_maxIter;
	
	public L2RParameter(String[] argv) {
		super(argv);
		
	}

}
