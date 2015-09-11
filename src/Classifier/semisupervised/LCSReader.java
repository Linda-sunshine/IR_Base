package Classifier.semisupervised;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import structures._Doc;
import structures._Pair;
import utils.Utils;

public class LCSReader implements Runnable{
	String m_filename;
	HashMap<_Pair, Integer> m_LCSMap;
	
	public LCSReader(String filename){
		m_filename = filename;
		m_LCSMap = new HashMap<_Pair, Integer>();
	}
	
	@Override
	public void run() {
		long start = System.currentTimeMillis();
		if (m_filename==null || m_filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(m_filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				String[] pair = line.split(",");
				
				if(pair.length != 3)
					System.out.println("Wrong LCS triple!");
				else
					m_LCSMap.put(new _Pair(Integer.parseInt(pair[0]), Integer.parseInt(pair[1].trim())), Integer.parseInt(pair[2].trim()));
			}
			reader.close();
			System.out.format("%d LCS triples loaded from %s in %.4f secs.\n", m_LCSMap.size(), m_filename, (double)(System.currentTimeMillis()-start)/1000.0);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", m_filename);
		}
	}		
	
	public HashMap<_Pair, Integer> getLCSMap(){
		return m_LCSMap;
	}
}
