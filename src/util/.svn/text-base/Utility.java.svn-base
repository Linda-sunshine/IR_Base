package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.TreeSet;
import java.util.Vector;

public class Utility {

	private static PorterStemmer stemmer = new PorterStemmer();
	
	public static double calculateCosineSimilarity(List<Double> set1, List<Double> set2)
	{
		double ret = 0;
		
		double [] s1 = new double[set1.size()];
		double [] s2 = new double[set2.size()];
		
		for ( int i = 0 ; i < s1.length ; i ++ )
			s1[i] = set1.get(i);
		for ( int i = 0 ; i < s2.length ; i ++ )
			s2[i] = set2.get(i);
		
		double upper = 0;
		for ( int i = 0 ; i < s1.length ; i ++ )
		{
			upper += s1[i] * s2[i];
		}
		
		double lower = 0;
		double magnitude1 = 0, magnitude2 = 0;

		for ( int i = 0 ; i < s1.length ; i ++ )
			magnitude1 += s1[i] * s1[i];
		magnitude1 = Math.sqrt(magnitude1);
		for ( int i = 0 ; i < s2.length ; i ++ )
			magnitude2 += s2[i] * s2[i];
		magnitude2 = Math.sqrt(magnitude2);
		
		lower = magnitude1 * magnitude2;
		
		ret = upper / lower;
		
		return ret;
	}
	
	public static void readNetworkStat(String networkStatFile, List<String> networkHeader, HashMap<String,List<Double>> networkMap) throws Exception
	{
		BufferedReader in = new BufferedReader(new FileReader(new File(networkStatFile)));
		
		String header = in.readLine();
		StringTokenizer headerTok = new StringTokenizer(header,",");
		headerTok.nextToken(); // ignore id 
		headerTok.nextToken(); // ignore topic
		while( headerTok.hasMoreTokens() == true )
		{
			networkHeader.add(headerTok.nextToken());
		}
		
		int cnt = 0;
		while(true)
		{
			String temp = in.readLine();
			if ( temp == null ) break;
			
			System.out.println("Reading Network Data Line : "+cnt+" : "+temp);
			cnt++;
			
			StringTokenizer tok = new StringTokenizer(temp,",");
			String ID = tok.nextToken();
			String URL = tok.nextToken();
			
			List<Double> valList = new ArrayList<Double>();
			while( tok.hasMoreTokens() == true )
				valList.add(new Double(tok.nextToken()));
			
			networkMap.put(ID, valList); 
		}
		
		in.close();
	}

	public static void readTopicStat(String topicStatFile, List<String> topicHeader, HashMap<String,List<Double>> topicMap) throws Exception
	{
		BufferedReader in = new BufferedReader(new FileReader(new File(topicStatFile)));
		
		int cnt = 0;
		while(true)
		{
			String temp = in.readLine();
			if ( temp == null ) break;
			
			System.out.println("Reading Topic Data Line : "+cnt+" : "+temp);
			
			StringTokenizer tok = new StringTokenizer(temp,",");
			String ID = tok.nextToken();
			String URL = tok.nextToken();
			
			List<Double> valList = new ArrayList<Double>();
			while( tok.hasMoreTokens() == true )
				valList.add(new Double(tok.nextToken()));
			
			if ( cnt == 0 )
			{
				for ( int i = 0 ; i < valList.size() ; i ++ )
					topicHeader.add( "Topic "+i );	
			}

			topicMap.put(ID, valList); 
			cnt++;
		}
		
		in.close();
	}

	public static void readNodeFile(String nodeFile, HashMap<String,Calendar> nodeTimeMap, HashMap<String,String> nodeMap) throws Exception
	{
		BufferedReader in = new BufferedReader(new FileReader(new File(nodeFile)));
		
		int cnt = 0;
		while(true)
		{
			String temp = in.readLine();
			if ( temp == null ) break;
			
			//System.out.println("Reading Network Node Data Line : "+cnt+" : "+temp);
			cnt++;
			
			StringTokenizer tok = new StringTokenizer(temp,",");
			String ID = tok.nextToken();
			String URL = tok.nextToken();
			String date = tok.nextToken();
			
			StringTokenizer dateTok = new StringTokenizer(date,"-");
			int y = (new Integer(dateTok.nextToken())).intValue();
			int m = (new Integer(dateTok.nextToken())).intValue();
			int d = (new Integer(dateTok.nextToken())).intValue();
			
			Calendar cal = Calendar.getInstance();
			cal.set(y,m,d);
			
			nodeTimeMap.put(ID,cal);
			nodeMap.put(URL, ID);
		}
		
		in.close();
	}

	public static void readLinkFile(String linkFile, List<String> sources, List<String> targets, HashMap<String,List<String>> neighborMap) throws Exception
	{
		BufferedReader in = new BufferedReader(new FileReader(new File(linkFile)));
		
		int cnt = 0;
		while(true)
		{
			String temp = in.readLine();
			if ( temp == null ) break;
			
//			System.out.println("Reading Network Link Data Line : "+cnt+" : "+temp);
			cnt++;
			
			StringTokenizer tok = new StringTokenizer(temp,",");
			String srcID = tok.nextToken();
			String tarID = tok.nextToken();
			
			sources.add(srcID);
			targets.add(tarID);
			
			List<String> neighbors = neighborMap.get(srcID);
			if ( neighbors == null ) neighbors = new ArrayList<String>();
			neighbors.add(tarID);
			neighborMap.remove(srcID);
			neighborMap.put(srcID, neighbors);
		}
		
		in.close();
	}
	
	public static void readImpactStat(String impactWidthFile, String impactDepthFile, String impactSizeFile, HashMap<String,List<Double>> impactMap) throws Exception
	{
		BufferedReader inWidth = new BufferedReader(new FileReader(new File(impactWidthFile)));
		BufferedReader inDepth = new BufferedReader(new FileReader(new File(impactDepthFile)));
		BufferedReader inSize = new BufferedReader(new FileReader(new File(impactSizeFile)));
		
		while( true )
		{
			String lineWidth = inWidth.readLine();
			String lineDepth = inDepth.readLine();
			String lineSize = inSize.readLine();
			
			if ( lineWidth == null ) break;
			
			StringTokenizer tokWidth = new StringTokenizer(lineWidth,",");
			StringTokenizer tokDepth = new StringTokenizer(lineDepth,",");
			StringTokenizer tokSize = new StringTokenizer(lineSize,",");
			
			String ID = tokWidth.nextToken();
			tokWidth.nextToken();
			tokDepth.nextToken(); tokDepth.nextToken();
			tokSize.nextToken(); tokSize.nextToken();
			
			List<Double> values = new ArrayList<Double>();
			values.add(	new Double(tokWidth.nextToken()) );
			values.add(	new Double(tokDepth.nextToken()) );
			values.add(	new Double(tokSize.nextToken()) );
			
			impactMap.put(ID, values);
		}
		
		inWidth.close();
		inDepth.close();
		inSize.close();
	}

	public static List<Double> normalizeTopicDistribution(List<Double> weights)
	{
		List<Double> ret = new ArrayList<Double>();
		
		double sum = 0;
		for ( int i = 0 ; i < weights.size() ; i ++ )
			sum += weights.get(i).doubleValue();
		
		for ( int i = 0 ; i < weights.size() ; i ++ )
			ret.add( weights.get(i).doubleValue() / sum );
		
		return ret;
	}
	
	public static double calculateMSEbetweenTopicDistributions(List<Double> t1, List<Double> t2)
	{
		double ret = 0;
		
		for ( int i = 0 ; i < t1.size() ; i ++ )
		{
			double error = t1.get(i).doubleValue() - t2.get(i).doubleValue();
			ret += error * error;
		}
	
		double size = t1.size();
		ret = ret / size;
		
		return ret;
	}

	public static int[] createRandomSequence(int size, int seed, int iteration) {
		int [] ret = new int[size];
		
		for ( int i = 0 ; i < ret.length ; i ++ )
			ret[i] = i;
		
		Random rand = new Random(seed);
		for ( int i = 0 ; i < iteration ; i ++ )
		{
			int src = rand.nextInt(size);
			int tar = rand.nextInt(size);
			
			int temp = ret[src];
			ret[src] = ret[tar];
			ret[tar] = temp;
		}
		
		return ret;
	}
	public static Vector<String> makeVectorFromFile(String fileName) throws Exception{
		Vector<String> list = new Vector<String>();
		String line;
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));

		while((line = reader.readLine()) != null){
			list.add(line.toLowerCase());
		}
		
		reader.close();
		return list;
	}
	
	
	public static Vector<String> makeStemmedVectorFromFile(String fileName, boolean usingStemmer) throws Exception {
		Vector<String> list = new Vector<String>();
		String line;
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));

		while((line = reader.readLine()) != null){
			if(usingStemmer==true)list.add(stemmer.stemming(line.toLowerCase()));
			else list.add(line.toLowerCase());
		}
		
		reader.close();
		return list;
	}

	public static Vector<String> makeStemmedValidWordFromFile(String fileName, boolean usingStemmer) throws Exception{
		
		Vector<String> list = new Vector<String>();
		String line;
		BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));

		while((line = reader.readLine()) != null){
			line = line.trim();
			/*
			if(line.contains("#")){
				line = line.split("#")[0];
			}
			*/
			
			if(usingStemmer == true)list.add(stemmer.stemming(line.toLowerCase()));
			else list.add(line.toLowerCase());
		}
		
		reader.close();
		return list;
	}
	
	public static TreeSet<String> makeSetOfWordsFromFile(String path, boolean usingStemmer) throws Exception {
		TreeSet<String> words = new TreeSet<String>();
		String line;
		BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
		while ((line = reader.readLine()) != null) {
//			if (usingStemmer == true) words.add(stemmer.stemming(line.toLowerCase()));
//			else words.add(line.toLowerCase());
			words.add(line);
		}
		return words;
	}
	

}
