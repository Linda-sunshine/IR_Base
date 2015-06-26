/**
 * 
 */
package Ranker.evaluator;

import structures._QUPair;
import structures._Query;

/**
 * @author wang296
 * 0. set up the query
 * 1. get the overall performance
 * 2. get performance change if swap Ui and Uj
 */
public interface Evaluation {
	public void setQuery(_Query query);
	
	public double eval(_Query query);
	
	public double delta(_QUPair qu1, _QUPair qu2);
	
	public void updateDeltas();
}
