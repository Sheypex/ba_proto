import my_yaml
import data_types
from pprint import pprint
import psycopg2
from db_actions import db_actions


def main():
    # test
    if False:
        try:
            db_actions.insert(data_types.node_configs_entry(12, 12, False), data_types.db_tables_enum.gcp_node_configs)
        except (Exception) as error:
            pprint(error)
        pprint(db_actions.select(data_types.node_configs_entry(num_vcpus=1, has_gpu=True),
                                 data_types.db_tables_enum.gcp_node_configs))
        pprint(db_actions.select(data_types.node_configs_entry(num_vcpus=1, has_gpu=True),
                                 data_types.db_tables_enum.gcp_node_configs, True))
        pprint(db_actions.select('*',
                                 data_types.db_tables_enum.gcp_node_configs))


if __name__ == '__main__':
    main()
