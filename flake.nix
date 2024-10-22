{
  description = "A poetry flake";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
  };

  outputs = {
    self,
    flake-utils,
    nixpkgs,
    poetry2nix,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      inherit (poetry2nix.lib.mkPoetry2Nix {inherit pkgs;}) mkPoetryApplication;
      rankwise = mkPoetryApplication {
        projectDir = self;
        meta = {
          name = "rankwise";
          mainProgram = "rankwise";
          description = "Rankwise";
          license = "Apache-2";
        };
      };
    in {
      formatter = pkgs.alejandra;

      devShells.default = pkgs.mkShell rec {
        packages = [
          pkgs.poetry
        ];
        buildInputs = [
          pkgs.zlib
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
        '';

      };

      packages.rankwise = rankwise;

      apps.default = {
        type = "app";
        program = "${rankwise}/bin/rankwise";
      };
    });
}
