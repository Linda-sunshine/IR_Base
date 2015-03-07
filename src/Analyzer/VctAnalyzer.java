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

	public VctAnalyzer(int classNo, int minDocLength) {
		super(classNo, minDocLength);
	}

	@Override
	public void LoadDoc(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			String[] container, entry;
			_SparseFeature[] spVct;
			_Doc doc;
			
			while ((line = reader.readLine()) != null) {
				container = line.split(" ");
				if (container.length<=m_lengthThreshold)
					continue;
				
				doc = new _Doc(m_corpus.getSize(), null, Integer.valueOf(container[0]));
				spVct = new _SparseFeature[container.length-1];
				for(int i=1; i<container.length; i++) {
					entry = container[i].split(":");
					spVct[i-1] = new _SparseFeature(Integer.valueOf(entry[0])-1, Double.valueOf(entry[1]));//the loaded feature index starts from 1
					
					//NOTE: do we need to update m_featureNames and m_featureNameIndex as well? We don't have it from the vector file
					if (!m_featureNameIndex.containsKey(entry[0])) { // fake feature name
						m_featureNameIndex.put(entry[0], m_featureNames.size());
						m_featureNames.add(Integer.toString(m_featureNames.size()));
					}
				}
				
				doc.setSpVct(spVct);
				m_corpus.addDoc(doc);
				m_classMemberNo[doc.getYLabel()]++;
			}
			reader.close();
			
			System.out.format("Loading %d vector files with %d features from %s...\n", m_corpus.getSize(), m_featureNames.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}

}
