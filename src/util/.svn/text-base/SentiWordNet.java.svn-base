package util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Vector;

public class SentiWordNet {

	private Vector<String> lemmas;
	private Hashtable<String, Double> pos;
	private Hashtable<String, Double> neg;

	public SentiWordNet(String sentiWordFile) {
		super();
		lemmas = new Vector<String>();
		pos = new Hashtable<String, Double>();
		neg = new Hashtable<String, Double>();
		try {
			// path of SentiWordNet
			String filename = sentiWordFile;
			BufferedReader file = new BufferedReader(new FileReader(filename));
			String input = file.readLine();
			while (input != null) {
				if (input.charAt(0) != '#') {
					String[] data = input.split("\t");

					// Note offset values can change in the same version of
					// WordNet due to minor edits in glosses.
					// Thus offsets are reported here just for reference, and
					// are not intended for use in applications.
					// Use lemmas instead.
					// String offset = data[0]+data[1];

					double positivity = Double.parseDouble(data[2]);
					double negativity = Double.parseDouble(data[3]);
					String[] synsetLemmas = data[4].split(" ");
					for (int i = 0; i < synsetLemmas.length; ++i) {
						String lemma = synsetLemmas[i].split("#")[0];
						pos.put(lemma, positivity);
						neg.put(lemma, negativity);
						lemmas.add(lemma);
					}
				}
				input = file.readLine();
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public double getPositivity(String lemma) {
		return pos.get(lemma);
	}

	public double getNegativity(String lemma) {
		return neg.get(lemma);
	}

	public double getObjetivity(String lemma) {
		return 1 - (pos.get(lemma) + neg.get(lemma));
	}

	public String[] getLemmas() {
		return lemmas.toArray(new String[0]);
	}
	
	public boolean containsKey(String lemma){
		return lemmas.contains(lemma);
	}
}
