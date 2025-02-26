provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "fraud_detection" {
  ami           = "ami-0c55b159cbfafe1f0" # Example AMI
  instance_type = "t2.micro"
}
