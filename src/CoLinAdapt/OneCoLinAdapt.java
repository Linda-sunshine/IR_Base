package CoLinAdapt;

import java.util.ArrayList;
import java.util.TreeMap;

import structures._User;

public class OneCoLinAdapt extends OneLinAdapt{

	public OneCoLinAdapt(_User u, int fg, int fn, TreeMap<Integer, ArrayList<Integer>> featureGroupIndex) {
		super(u, fg, fn, featureGroupIndex);
	}

}
