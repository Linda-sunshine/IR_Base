package util;

public class LinkNode<T> extends Node<T> {
	
	public LinkNode() {
		super();
		addLink(null);  // Prev
		addLink(null);  // Next
	}
	
	public LinkNode(T value) {
		this();
		setValue(value);
	}
	
	public LinkNode<T> getPrevNode() {
		return (LinkNode<T>) get(0);
	}
	
	public LinkNode<T> getNextNode() {
		return (LinkNode<T>) get(1);
	}
	
	public void setPrevNode(LinkNode<T> node) {
		set(0, node);
	}
	
	public void setNextNode(LinkNode<T> node) {
		set(1, node);
	}
}