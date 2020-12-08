from milvus import Milvus, MetricType
import torch


class SearchResult:
    def __init__(self, data):
        self.data = data

    def id(self, device):
        return torch.tensor(self.data.id_array).to(device)

    def dist(self, device):
        return torch.tensor(self.data.distance_array).to(device)


class MilvusConnection:
    def __init__(self, env, name="movies_L2", port="19530", param=None):

        if param is None:
            param = dict()
        param = {
            "collection_name": name,
            "dimension": 128,
            "index_file_size": 1024,
            "metric_type": MetricType.L2,
            **param,
        }
        self.name = name
        self.client = Milvus(host="localhost", port=port)
        self.statuses = {}
        if not self.client.has_collection(name)[1]:
            status_created_collection = self.client.create_collection(param)
            vectors = env.base.embeddings.detach().cpu().numpy().astype("float32")
            target_ids = list(range(vectors.shape[0]))
            status_inserted, inserted_vector_ids = self.client.insert(
                collection_name=name, records=vectors, ids=target_ids
            )
            status_flushed = self.client.flush([name])
            status_compacted = self.client.compact(collection_name=name)
            self.statuses["created_collection"] = status_created_collection
            self.statuses["inserted"] = status_inserted
            self.statuses["flushed"] = status_flushed
            self.statuses["compacted"] = status_compacted

    def search(self, search_vecs, topk=10, search_param=None):
        if search_param is None:
            search_param = dict()
        search_param = {"nprobe": 16, **search_param}
        status, results = self.client.search(
            collection_name=self.name,
            query_records=search_vecs,
            top_k=topk,
            params=search_param,
        )
        self.statuses['last_search'] = status
        return SearchResult(results)

    def get_log(self):
        return self.statuses
