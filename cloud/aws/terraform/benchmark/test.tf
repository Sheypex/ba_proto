terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region                  = "eu-central-1"
  shared_credentials_file = "$HOME/.aws/credentials"
}

resource "aws_vpc" "dflt" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_security_group" "ssh-in" {
  name   = "ssh-in"
  vpc_id = aws_vpc.dflt.id

  ingress {
    cidr_blocks = ["0.0.0.0/0"]
    from_port   = 22
    protocol    = "tcp"
    to_port     = 22
  }
}

resource "aws_security_group" "all-egress" {
  name   = "all-egress"
  vpc_id = aws_vpc.dflt.id

  egress {
    cidr_blocks = ["0.0.0.0/0"]
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
  }
}

resource "aws_security_group" "all-internal-ingress" {
  name   = "all-internal-ingress"
  vpc_id = aws_vpc.dflt.id

  ingress {
    cidr_blocks = [aws_vpc.dflt.cidr_block]
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
  }
}

resource "aws_subnet" "dflt" {
  vpc_id                  = aws_vpc.dflt.id
  cidr_block              = "10.0.0.0/24"
  availability_zone       = "eu-central-1a"
  map_public_ip_on_launch = true
}

resource "aws_internet_gateway" "dflt" {
  vpc_id = aws_vpc.dflt.id
}

resource "aws_route_table" "dflt" {
  vpc_id = aws_vpc.dflt.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.dflt.id
  }
}

resource "aws_key_pair" "dflt" {
  key_name   = "dflt"
  public_key = file("../../id_rsa.pub")
}

resource "aws_route_table_association" "dflt" {
  subnet_id      = aws_subnet.dflt.id
  route_table_id = aws_route_table.dflt.id
}

data "aws_ami" "ubuntu_x86" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

data "aws_ami" "ubuntu_arm" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-arm64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

locals {
  ec2 = yamldecode(file("../../node_configs.yaml"))
}

variable "instances_to_launch" {
  type    = list(list(string))
  default = [["M", "m6g", "large"]]
  validation {
    condition     = length(var.instances_to_launch) > 0 && alltrue([for entry in var.instances_to_launch : length(entry) == 3])
    error_message = "Needs exactly 3 values: ['family', 'type', 'size']."
  }
}

resource "aws_instance" "testNode" {
  for_each = { for index, entry in var.instances_to_launch : join("", [local.ec2[entry[0]][entry[1]].size[entry[2]].name, "-", index]) => {
    #family         = entry[0]
    #type           = entry[1]
    #size           = entry[2]
    instance_model = local.ec2[entry[0]][entry[1]]
    instance_type  = local.ec2[entry[0]][entry[1]].size[entry[2]]
    }
  }
  ami           = each.value.instance_model.arch == "arm" ? data.aws_ami.ubuntu_arm.id : data.aws_ami.ubuntu_x86.id
  instance_type = each.value.instance_type.name

  subnet_id       = aws_subnet.dflt.id
  vpc_security_group_ids = [aws_security_group.ssh-in.id, aws_security_group.all-egress.id, aws_security_group.all-internal-ingress.id]

  root_block_device {
    volume_size = 120
    volume_type = "gp3"
  }

  key_name = aws_key_pair.dflt.id
  tags = {
    "name"         = each.value.instance_type.name
    "vcpu"         = each.value.instance_type.vcpus
    "gpu"          = each.value.instance_type.gpu
    "ram"          = each.value.instance_type.ram
    "storage_type" = "ebs" #each.value.instance_type.storage_type
    "disk_type"    = "ssd" #each.value.instance_type.disk_type
    "iperf"        = false
    "testing"      = true
    "index"        = split("-", each.key)[1]
  }
}

# output "testNodes" {
#   value = aws_instance.testNode
# }
