/**
 * 
 */
package structures;

import java.util.ArrayList;

/**
 * @author hongning
 * A node in random walk graph
 */
public class _Node {

	public ArrayList<_Edge> m_labeledEdges; // edge to the labeled neighbors
	public ArrayList<_Edge> m_unlabeledEdges; // edge to the unlabeled neighbors
	public double m_label; // ground-truth label
	public double m_pred; // predicted label (assigned by random walk)
	public double m_classifierPred; // classifier's prediction (assigned by classifier)
	
	public _Node(double label, double classifier) {
		m_labeledEdges = null;
		m_unlabeledEdges = null;
		m_label = label;
		m_classifierPred = classifier;
		m_pred = classifier; // start from classifier's prediction
	}
	
	public _Node(double label) {
		m_labeledEdges = null;
		m_unlabeledEdges = null;
		m_label = label;
		m_classifierPred = label;
		m_pred = label;
	}

	public void addLabeledEdge(_Node n, double sim) {
		if (m_labeledEdges == null)
			m_labeledEdges = new ArrayList<_Edge>();
		
		m_labeledEdges.add(new _Edge(n, sim));
	}
	
	public void addUnlabeledEdge(_Node n, double sim) {
		if (m_unlabeledEdges == null)
			m_unlabeledEdges = new ArrayList<_Edge>();
		
		m_unlabeledEdges.add(new _Edge(n, sim));
	}
}

