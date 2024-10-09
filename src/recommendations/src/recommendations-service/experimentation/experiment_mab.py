# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
import logging
from datetime import datetime
from typing import Dict
import boto3  # Added import statement

from botocore.exceptions import ClientError

from experimentation.experiment import BuiltInExperiment

log = logging.getLogger(__name__)

class MultiArmedBanditExperiment(BuiltInExperiment):
    """ Implementation of the multi-armed bandit problem using the Thompson Sampling approach
    to exploring variations to identify and exploit the best performing variation
    """
    def __init__(self, table, **data):
        super().__init__(table, **data)
        # Initialize DynamoDB resource
        self.dynamodb = boto3.resource('dynamodb')
        self.log_table = self.dynamodb.Table('experiment_logs')

    def get_items(self, user_id, current_item_id=None, item_list=None, num_results=10,
                  tracker=None, filter_values=None, context=None, timestamp: datetime = None,
                  promotion: Dict = None):
        if not user_id:
            raise Exception('user_id is required')
        if len(self.variations) < 2:
            raise Exception(f'Experiment {self.id} does not have 2 or more variations')

        # Determine the variation to use.
        variation_idx = self._select_variation_index()
        log.debug(f'{self._getClassName()} - assigned user {user_id} to variation {variation_idx} for experiment {self.feature}.{self.name}')

        # Increment exposure count for variation
        self._increment_exposure_count(variation_idx)

        # Fetch recommendations using the variation's resolver
        variation = self.variations[variation_idx]

        resolve_params = {
            'user_id': user_id,
            'product_id': current_item_id,
            'product_list': item_list,
            'num_results': num_results,
            'filter_values': filter_values,
            'context': context,
            'promotion': promotion
        }
        items = variation.resolver.get_items(**resolve_params)

        # Inject experiment details into recommended items list
        rank = 1
        for item in items:
            correlation_id = self._create_correlation_id(user_id, variation_idx, rank)

            item_experiment = {
                'id': self.id,
                'feature': self.feature,
                'name': self.name,
                'type': self.type,
                'variationIndex': variation_idx,
                'resultRank': rank,
                'correlationId': correlation_id
            }
            item.update({
                'experiment': item_experiment
            })
            rank += 1

        if tracker is not None:
            # Track exposure details for analysis
            timestamp = datetime.now() if not timestamp else timestamp
            event = {
                'event_type': 'Experiment Exposure',
                'event_timestamp': int(round(timestamp.timestamp() * 1000)),
                'attributes': {
                    'user_id': user_id,
                    'experiment': {
                        'id': self.id,
                        'feature': self.feature,
                        'name': self.name,
                        'type': self.type
                    },
                    'variation_index': variation_idx,
                    'variation': variation.config
                }
            }

            tracker.log_exposure(event)

        return items

    def _select_variation_index(self):
        """ Selects the variation using Thompson Sampling """
        variation_count = len(self.variations)
        exposures = np.zeros(variation_count)
        conversions = np.zeros(variation_count)

        for i in range(variation_count):
            variation = self.variations[i]
            exposures[i] = int(variation.config.get('exposures', 0))
            conversions[i] = int(variation.config.get('conversions', 0))

        # Sample from posterior (this is the Thompson Sampling approach)
        theta = np.random.beta(conversions + 1, exposures + 1)

        try:
            self._log_experiment_data(exposures, conversions, theta)
        except Exception as e:
            log.error(f"Error logging experiment data: {e}")

        # Select variation index with highest posterior probability of converting
        return np.argmax(theta)

    def _log_experiment_data(self, exposures, conversions, theta):
        timestamp = datetime.utcnow().isoformat()
        item = {
            'experiment_id': self.id,
            'timestamp': timestamp,
            'exposures': exposures.tolist(),
            'conversions': conversions.tolist(),
            'theta': theta.tolist()
        }
        try:
            self.log_table.put_item(Item=item)
            log.debug(f"Logged experiment data at {timestamp}")
        except ClientError as e:
            log.error(f"Failed to log experiment data: {e}")
        except Exception as e:
            log.error(f"Unexpected error during the logging: {e}")
