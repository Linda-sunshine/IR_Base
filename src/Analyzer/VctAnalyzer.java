/**
 * 
 */
package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import structures._Doc;
import structures._SparseFeature;

/**
 * @author hongning
 * loading the vector format of text documents
 */
public class VctAnalyzer extends Analyzer {

	public VctAnalyzer(int classNo, int minDocLength, String providedCV) {
		super(classNo, minDocLength);
		m_isCVLoaded = LoadCV(providedCV);
	}

	@Override
	public void LoadDoc(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			String[] container, entry;
			_SparseFeature[] spVct;
			_Doc doc;
			int maxFvIndex = 0, index;
			
			while ((line = reader.readLine()) != null) {
				container = line.split(" ");
				if (container.length<=m_lengthThreshold) //  || Math.random() < 0.65
					continue;
				
				doc = new _Doc(m_corpus.getSize(), null, Integer.valueOf(container[0]));
				if (!line.contains("#"))
					spVct = new _SparseFeature[container.length-1];
				else
					spVct = new _SparseFeature[container.length-2]; // exclude the comment
				
				for(int i=1; i<container.length; i++) {
					if (container[i].startsWith("#")) {//parse the comment part for this review
						entry = container[i].split("-");
						doc.setItemID(entry[0].substring(1));
						doc.setName(entry[1]);
					} else {
						entry = container[i].split(":");
						index = Integer.valueOf(entry[0])-1;//the loaded feature index starts from 1
						spVct[i-1] = new _SparseFeature(index, Double.valueOf(entry[1]));
						
						if (index>maxFvIndex)
							maxFvIndex = index;
					}
				}
				
				doc.setSpVct(spVct);
				m_corpus.addDoc(doc);
				m_classMemberNo[doc.getYLabel()]++;
			}
			reader.close();
			reviseCV(maxFvIndex);
			
			System.out.format("Loading %d vector files with %d features from %s...\n", m_corpus.getSize(), m_featureNames.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	void reviseCV(int maxFvIndex) {
		if (1+maxFvIndex == m_featureNames.size())
			return; // the size matches, no need to revise
		else if (maxFvIndex >= m_featureNames.size()) {
			System.err.format("The loaded CV list has less indexed features (%d) than those in the vector file (%d)! Exit!\n", m_featureNames.size(), maxFvIndex+1);
			System.exit(-1);
		} else {
			System.err.format("The loaded CV list has more indexed features (%d) than those in the vector file (%d)! Trim it!", m_featureNames.size(), maxFvIndex+1);
		}
		
		//if we have fewer features in the vector file than those in the CV list, trim the CV list
		String feature;
		while (m_featureNames.size()>1+maxFvIndex) {
			feature = m_featureNames.remove(1+maxFvIndex);
			m_featureStat.remove(feature);
			m_featureNameIndex.remove(feature);
		}
	}
}
