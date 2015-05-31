/******Analyzer.java******************/
//add one more hashMap for finding features with index.
protected HashMap<Integer, String> m_featureIndexName; //key: index of the feature; value: content of the feature 

//Initialize it in constructor function.
public Analyzer(int classNo, int minDocLength) {
	m_featureIndexName = new HashMap<Integer, String>();
}

protected boolean LoadCV(String filename) {
	if (filename==null || filename.isEmpty())
		return false;
	
	try {
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
		String line;
		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#")){
				if (line.startsWith("#NGram")) {//has to be decoded
					int pos = line.indexOf(':');
					m_Ngram = Integer.valueOf(line.substring(pos+1));
				}
					
			} else 
				expandVocabulary(line);
		}
		reader.close();
		
		System.out.format("%d feature words loaded from %s...\n", m_featureNames.size(), filename);
		m_isCVLoaded = true;
		/************If CV is loaded, set the look-up table.***********/
		setFeatureIndexName();
		
		return true;
	} catch (IOException e) {
		System.err.format("[Error]Failed to open file %s!!", filename);
		return false;
	}
}

//Build the hashMap: given the index, find the corresponding feature.
public void setFeatureIndexName(){
	m_featureIndexName.clear();
	for(String name: m_featureNameIndex.keySet()){
		m_featureIndexName.put(m_featureNameIndex.get(name), name);
	}
}


/*********DocAnalyzer.java**************/
public void rollBack(HashMap<Integer, Double> spVct, int y){
	if (!m_isCVLoaded) {
		setFeatureIndexName();//Get the current index-feature lookup table.
		for(int index: spVct.keySet()){
			String token = m_featureIndexName.get(index);
			_stat stat = m_featureStat.get(token);
			if(Utils.sumOfArray(stat.getDF())==1){//If the feature is the first time to show in feature set.
				m_featureNameIndex.remove(index);
				m_featureStat.remove(token);
				m_featureNames.remove(token);
			}
			else{//If the feature is not the first time to show in feature set.
				m_featureStat.get(token).minusOneDF(y);
				m_featureStat.get(token).minusNTTF(y, spVct.get(index));
			}
		}
	} else{//If CV is loaded, we can minus the DF and TTF directly.
//		setFeatureIndexName(); //Add this step to LoadCV to avoid repetitive operations.
		for(int index: spVct.keySet()){
			String token = m_featureIndexName.get(index);
			m_featureStat.get(token).minusOneDF(y);
			m_featureStat.get(token).minusNTTF(y, spVct.get(index));
		}
	}
}

protected boolean AnalyzeDoc(_Doc doc) {
	String[] tokens = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
	int y = doc.getYLabel();
	// Construct the sparse vector.
	HashMap<Integer, Double> spVct = constructSpVct(tokens, y);
	
	if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose
		doc.createSpVct(spVct);
		m_corpus.addDoc(doc);
		m_classMemberNo[y]++;
		if (m_releaseContent)
			doc.clearSource();
		return true;
	} else{
		/****Roll back here!!******/
		rollBack(spVct, y);
		return false;
	}
}

/************_stat.java******************/
public void minusOneDF(int index){
	this.m_DF[index]--;
}

public void minusNTTF(int index, double n){
	this.m_TTF[index] -= n;
}

