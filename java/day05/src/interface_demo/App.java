package interface_demo;

public class App {
    Shape rec = new Rectangle();
}

 interface Shape{

    public float CalcArea();
    public float CalcPerimeter();

}


class Rectangle implements Shape{
    float length,breadth;

    public Rectangle() {
        super();
        this.length=0;
        this.breadth=0;
    }
    public Rectangle(float length,float breadth) {
        super();
        this.length = length;
        this.breadth = breadth;
    }

    @Override
    public float CalcArea() {
        float area=0;
        area = this.length*this.breadth;
        return area;
    }

    @Override
    public float CalcPerimeter() {
        float perimeter=0;
        perimeter = 2*(this.length+this.breadth);
        return perimeter;
    }

}


class Square extends Rectangle{
    float side;
    public Square() {
        super(0,0);
    }

    public Square(float side) {
        super(side,side);
    }

    @Override
    public float CalcArea() {
        return super.CalcArea();
    }

    @Override
    public float CalcPerimeter() {
        return super.CalcPerimeter();
    }
}


class Circle implements Shape{
    float radius;

    public Circle() {
        this.radius = 0;
    }    
    public Circle(float radius) {
        this.radius = radius;
    }

    @Override
    public float CalcArea() {
        float area=0;
        area = (22.0f/7)*radius*radius;
        return area;
    }

    @Override
    public float CalcPerimeter() {
        float perimeter=0;
        perimeter = 2.0f*(22.0f/7)*radius;
        return perimeter;
    }
    
}
