# -*- coding: utf-8 -*-

# this class is not currently used and is therefore not under version control

class Filiale:
    def __init__(self, x, y, name, hektar_information, logger):
        self.x = x
        self.y = y
        self.name = name
        self.hektar_information = hektar_information
        self.logger = logger
        # constants
        self.HA = 7800.00 # Haushaltausgabe

    def calculate_umsatzpotential(self):
        sum = 0.0
        for hektar in self.hektar_information:
            (marktanteil, anzahl_haushalte) = self.hektar_information[hektar]
            sum += marktanteil*anzahl_haushalte

        sum *= self.HA
        return sum

    def update_hektar_info(self, hektar_id, marktanteil, anzahl_haushalte):
        if hektar_id not in self.hektar_information:
            self.logger.error("update_hektar_info: hektar_id %s has not been associated with Filiale %s."
                              " This will be ignored, but must be nevertheless investigated.",
                              hektar_id, self.name)
        else:
            if self.hektar_information[hektar_id] != (0, 0):
                self.logger.warning("update_hektar_info: hektar_id %s has already been updated "
                                    "for Filiale %s. Ignoring", hektar_id, self.name)
            else:
                self.hektar_information[hektar_id] = (marktanteil, anzahl_haushalte)

    def print(self):
        print("FILIALE NAME: %s \t X:%d \t Y:%d. Hektar information follows:")
        for h in self.hektar_information:
            print("\t Hektar id / Marktanteil =  %s / %f", h, self.hektar_information[h])

