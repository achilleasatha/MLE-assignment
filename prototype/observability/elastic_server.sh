docker network create elastic
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.12.2
sudo sysctl -w vm.max_map_count=262144
docker run --name es01 --net elastic -p 9200:9200 -it -m 1GB docker.elastic.co/elasticsearch/elasticsearch:8.12.2
