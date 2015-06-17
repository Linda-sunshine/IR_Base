package util;

import java.util.ArrayList;

public class Node<T> {
	private ArrayList<Node<T>> links;
	T value;
	
	public Node() {
		this.links = new ArrayList<Node<T>>();
		this.value = null;
	}
	
	public void addLink(Node<T> node) {
		links.add(node);
	}
	
	public void set(int index, Node<T> node) {
		links.set(index, node);
	}
	
	public Node<T> get(int key) {
		return links.get(key);
	}
	
	public Node<T> get(Iterable<Integer> key) {
		Node<T> node = this;
		for (int index : key) {
			node = node.links.get(index);
		}
		return node;
	}
	
	public T getValue(Iterable<Integer> key) {
		return get(key).value;
	}
	
	public T getValue(int key) {
		return links.get(key).value;
	}
	
	public T getValue() {
		return value;
	}
	
	public T setValue(Iterable<Integer> key, T value) {
		return get(key).setValue(value);
	}
	
	public T setValue(int key, T value) {
		return links.get(key).setValue(value);
	}
	
	public T setValue(T value) {
		this.value = value;
		return this.value;
	}
	
}
